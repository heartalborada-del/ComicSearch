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
