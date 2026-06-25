/**
 * Auth API module.
 * Wraps auth endpoints for login, register, status, and quota.
 */
import { getJson, postJson } from './client'
import type {
    AuthStatusResponse,
    LoginRequest,
    QuotaResponse,
    RegisterRequest,
    TokenResponse,
    UserInfo,
} from '@/types/auth'

/**
 * Get the current auth status (enabled, login state, user info).
 */
export function getAuthStatus(): Promise<AuthStatusResponse> {
    return getJson<AuthStatusResponse>('/auth/status')
}

/**
 * Register a new user account.
 */
export function registerUser(data: RegisterRequest): Promise<TokenResponse> {
    return postJson<TokenResponse>('/auth/register', data)
}

/**
 * Login with username and password.
 */
export function loginUser(data: LoginRequest): Promise<TokenResponse> {
    return postJson<TokenResponse>('/auth/login', data)
}

/**
 * Get current logged-in user info.
 */
export function getMe(): Promise<UserInfo> {
    return getJson<UserInfo>('/auth/me')
}

/**
 * Get current user's search quota.
 */
export function getQuota(): Promise<QuotaResponse> {
    return getJson<QuotaResponse>('/auth/quota')
}

/** Request body for POST /auth/quota/set (admin only). */
export interface SetQuotaRequest {
    user_id: number
    daily_quota: number
}

/** Response from POST /auth/quota/set. */
export interface SetQuotaResponse {
    user_id: number
    username: string
    daily_quota: number | null
}

/**
 * Admin: set per-user daily search quota.
 */
export function setUserQuota(data: SetQuotaRequest): Promise<SetQuotaResponse> {
    return postJson<SetQuotaResponse>('/auth/quota/set', data)
}

/** Request body for POST /auth/users/reset-password (admin only). */
export interface ResetPasswordRequest {
    user_id: number
    new_password: string
}

/** Response from POST /auth/users/reset-password. */
export interface ResetPasswordResponse {
    user_id: number
    username: string
}

/**
 * Admin: reset a user's password.
 */
export function resetUserPassword(data: ResetPasswordRequest): Promise<ResetPasswordResponse> {
    return postJson<ResetPasswordResponse>('/auth/users/reset-password', data)
}
