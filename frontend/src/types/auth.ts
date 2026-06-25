/**
 * Type definitions for the auth API.
 */

/** User info returned by GET /auth/me and in login/register response. */
export interface UserInfo {
    id: number
    username: string
    is_admin: boolean
    created_at: string
}

/** Response from POST /auth/login and POST /auth/register. */
export interface TokenResponse {
    access_token: string
    token_type: string
    user: UserInfo
}

/** Response from GET /auth/status. */
export interface AuthStatusResponse {
    auth_enabled: boolean
    turnstile_site_key: string | null
    logged_in: boolean
    user: UserInfo | null
}

/** Response from GET /auth/quota. */
export interface QuotaResponse {
    auth_enabled: boolean
    daily_quota: number
    used_today: number
    remaining: number
    is_admin: boolean
    quota_reset_at: string | null
}

/** Request body for POST /auth/register. */
export interface RegisterRequest {
    username: string
    password: string
    turnstile_token?: string
}

/** Request body for POST /auth/login. */
export interface LoginRequest {
    username: string
    password: string
    turnstile_token?: string
}
