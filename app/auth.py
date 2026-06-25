from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import httpx
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import AuthSettings
from app.db import get_db
from app.models import SearchUsage, User

logger = logging.getLogger("uvicorn.error")

ALGORITHM = "HS256"
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
IP_QUOTA_MULTIPLIER = 10

_bearer_scheme = HTTPBearer(auto_error=False)

# ---- In-memory IP usage tracker (not persisted) ----
_ip_usage: dict[str, dict[str, int]] = {}  # {date: {ip: count}}


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: int, username: str, settings: AuthSettings) -> str:
    """Create a JWT access token for the given user."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.token_expire_minutes)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "username": username,
        "exp": expire,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str, settings: AuthSettings) -> dict[str, Any]:
    """Decode and validate a JWT access token."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or expired token",
        ) from exc


async def verify_turnstile_token(
    token: str,
    settings: AuthSettings,
    remote_ip: str | None = None,
) -> bool:
    """Verify a Cloudflare Turnstile token via the siteverify API.

    Returns True if verification succeeds. If no secret key is configured,
    verification is skipped (returns True).
    """
    if settings.turnstile_secret_key is None:
        return True

    form_data: dict[str, str] = {
        "secret": settings.turnstile_secret_key,
        "response": token,
    }
    if remote_ip:
        form_data["remoteip"] = remote_ip

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(TURNSTILE_VERIFY_URL, data=form_data)
            resp.raise_for_status()
            result = resp.json()
    except Exception as exc:
        logger.error("turnstile verification request failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="captcha verification service unavailable",
        ) from exc

    if not result.get("success", False):
        error_codes = result.get("error-codes", [])
        logger.warning("turnstile verification failed: %s", error_codes)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"captcha verification failed: {', '.join(error_codes) or 'unknown error'}",
        )

    return True


def _today_str() -> str:
    """Return today's date as YYYY-MM-DD string (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _quota_reset_at() -> str:
    """Return the ISO timestamp of the next UTC midnight (quota reset time)."""
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return tomorrow.isoformat()


def _get_user_quota(user: User, settings: AuthSettings) -> int:
    """Get the effective daily quota for a user.

    Priority: user.daily_quota (if set) > settings.daily_search_quota > 0 (unlimited).
    """
    if user.daily_quota is not None:
        return user.daily_quota
    return settings.daily_search_quota


def _check_ip_quota(request: Request, settings: AuthSettings) -> None:
    """Enforce per-IP daily limit (user_quota * 10).

    Only applies when auth is enabled and user is not authenticated.
    """
    if not settings.enabled:
        return
    if settings.daily_search_quota <= 0:
        return

    client_ip = request.client.host if request.client else None
    if client_ip is None:
        return

    ip_limit = settings.daily_search_quota * IP_QUOTA_MULTIPLIER
    today = _today_str()

    # Clean stale entries
    stale = [d for d in _ip_usage if d != today]
    for d in stale:
        del _ip_usage[d]

    day_map = _ip_usage.setdefault(today, {})
    count = day_map.get(client_ip, 0)
    if count >= ip_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"IP daily search quota exceeded ({ip_limit}/day)",
        )


def _increment_ip_quota(request: Request, settings: AuthSettings) -> None:
    """Increment the per-IP daily usage counter."""
    if not settings.enabled:
        return
    client_ip = request.client.host if request.client else None
    if client_ip is None:
        return
    today = _today_str()
    _ip_usage.setdefault(today, {})[client_ip] = _ip_usage.setdefault(today, {}).get(client_ip, 0) + 1


def get_today_usage(db: Session, user_id: int) -> SearchUsage:
    """Get or create today's search usage record for a user."""
    today = _today_str()
    record = db.execute(
        select(SearchUsage).where(
            SearchUsage.user_id == user_id,
            SearchUsage.usage_date == today,
        )
    ).scalar_one_or_none()

    if record is None:
        record = SearchUsage(user_id=user_id, usage_date=today, count=0)
        db.add(record)
        db.commit()
        db.refresh(record)

    return record


def get_current_user_optional(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    """Get the current user from the Authorization header.

    Returns None if no credentials are provided. Raises 401 if credentials
    are invalid.
    """
    settings = request.app.state.runtime.settings
    if not settings.auth.enabled:
        return None

    if credentials is None:
        return None

    payload = decode_access_token(credentials.credentials, settings.auth)
    user_id = int(payload["sub"])
    user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()

    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="user not found or inactive",
        )

    return user


def require_auth(
    request: Request,
    user: User | None = Depends(get_current_user_optional),
) -> User:
    """Dependency that requires authentication when auth is enabled.

    When auth is disabled, this dependency passes through without a user.
    """
    settings = request.app.state.runtime.settings
    if not settings.auth.enabled:
        # Auth disabled — return a dummy user-like object
        return User(  # type: ignore[call-arg]
            id=0,
            username="anonymous",
            password_hash="",
            is_active=True,
            is_admin=True,
            created_at="",
        )

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication required",
        )

    return user


def require_admin(
    request: Request,
    user: User = Depends(require_auth),
) -> User:
    """Dependency that requires the current user to be an admin.

    When auth is disabled the dummy admin user is always returned.
    """
    settings = request.app.state.runtime.settings
    if not settings.auth.enabled:
        return user

    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin privileges required",
        )

    return user


def require_search_quota(
    request: Request,
    user: User = Depends(require_auth),
    db: Session = Depends(get_db),
) -> User:
    """Dependency that checks the user's daily search quota + per-IP limit.

    - Authenticated users: checked against their effective daily quota
      (user.daily_quota or the global default).
    - Unauthenticated (auth disabled): IP-based limit applies.
    - Admins are exempt.
    """
    settings = request.app.state.runtime.settings
    if not settings.auth.enabled:
        # Auth disabled — use IP-based limiting only
        _check_ip_quota(request, settings)
        return user

    if user.is_admin:
        return user

    # Per-IP quota check (always, for both auth and non-auth)
    _check_ip_quota(request, settings)

    quota = _get_user_quota(user, settings)
    if quota <= 0:
        return user  # unlimited

    usage = get_today_usage(db, user.id)
    if usage.count >= quota:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"daily search quota exceeded ({quota}/day)",
        )

    return user


def consume_search_quota(
    user: User,
    db: Session,
    settings: AuthSettings,
    request: Request | None = None,
) -> None:
    """Increment the user's daily search usage counter + IP counter."""
    if request is not None:
        _increment_ip_quota(request, settings)

    if not settings.enabled:
        return
    if user.is_admin:
        return

    usage = get_today_usage(db, user.id)
    usage.count += 1
    db.commit()


def get_quota_info(
    user: User,
    db: Session,
    settings: AuthSettings,
) -> dict[str, Any]:
    """Return quota information for the current user."""
    reset_at = _quota_reset_at()

    if not settings.enabled:
        return {
            "auth_enabled": False,
            "daily_quota": 0,
            "used_today": 0,
            "remaining": 0,
            "is_admin": True,
            "quota_reset_at": reset_at,
        }

    if user.is_admin:
        return {
            "auth_enabled": True,
            "daily_quota": -1,
            "used_today": 0,
            "remaining": -1,
            "is_admin": True,
            "quota_reset_at": reset_at,
        }

    quota = _get_user_quota(user, settings)
    usage = get_today_usage(db, user.id)
    remaining = max(0, quota - usage.count) if quota > 0 else -1

    return {
        "auth_enabled": True,
        "daily_quota": quota,
        "used_today": usage.count,
        "remaining": remaining,
        "is_admin": False,
        "quota_reset_at": reset_at,
    }