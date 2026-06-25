/**
 * Auth store — manages login state, token persistence, quota tracking.
 */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { getAuthStatus, getMe, getQuota, loginUser, registerUser } from '@/api/auth'
import { ApiError } from '@/api/client'
import { setAuthToken } from '@/api/client'
import type { QuotaResponse, UserInfo } from '@/types/auth'

const TOKEN_KEY = 'comicsearch-token'

export const useAuthStore = defineStore('auth', () => {
    const authEnabled = ref(false)
    const turnstileSiteKey = ref<string | null>(null)
    const loggedIn = ref(false)
    const user = ref<UserInfo | null>(null)
    const token = ref<string | null>(null)
    const loading = ref(false)
    const loginLoading = ref(false)
    const loginError = ref<string | null>(null)
    const quota = ref<QuotaResponse | null>(null)

    const isAdmin = computed(() => user.value?.is_admin ?? false)
    const quotaRemaining = computed(() => quota.value?.remaining ?? 0)
    const quotaUsed = computed(() => quota.value?.used_today ?? 0)
    const quotaTotal = computed(() => quota.value?.daily_quota ?? 0)
    const quotaUnlimited = computed(() => quota.value?.is_admin === true || quota.value?.remaining === -1)

    /**
     * Check auth status from server and load persisted token.
     */
    async function checkAuthStatus(): Promise<void> {
        loading.value = true
        try {
            const status = await getAuthStatus()
            authEnabled.value = status.auth_enabled
            turnstileSiteKey.value = status.turnstile_site_key

            if (status.logged_in && status.user) {
                loggedIn.value = true
                user.value = status.user
            } else {
                // Try to restore from localStorage if server says not logged in
                // (e.g. token expired on server side)
                const stored = localStorage.getItem(TOKEN_KEY)
                if (stored && status.auth_enabled) {
                    try {
                        // Try to refresh user info with stored token
                        setAuthToken(stored)
                        const userInfo = await getMe()
                        loggedIn.value = true
                        user.value = userInfo
                        token.value = stored
                    } catch {
                        // Token expired or invalid
                        clearAuth()
                    }
                }
            }
        } catch {
            // Auth API might not be available if auth is disabled
            authEnabled.value = false
        } finally {
            loading.value = false
        }
    }

    /**
     * Fetch the current quota status.
     */
    async function fetchQuota(): Promise<void> {
        if (!loggedIn.value) return
        try {
            quota.value = await getQuota()
        } catch {
            // Silently ignore quota fetch errors
        }
    }

    /**
     * Register a new account.
     */
    async function register(username: string, password: string, turnstileToken?: string): Promise<void> {
        loginLoading.value = true
        loginError.value = null
        try {
            const response = await registerUser({
                username,
                password,
                turnstile_token: turnstileToken,
            })
            handleLoginResponse(response)
        } catch (err) {
            loginError.value = err instanceof ApiError ? err.detail : '注册失败'
            throw err
        } finally {
            loginLoading.value = false
        }
    }

    /**
     * Login with username and password.
     */
    async function login(username: string, password: string, turnstileToken?: string): Promise<void> {
        loginLoading.value = true
        loginError.value = null
        try {
            const response = await loginUser({
                username,
                password,
                turnstile_token: turnstileToken,
            })
            handleLoginResponse(response)
        } catch (err) {
            loginError.value = err instanceof ApiError ? err.detail : '登录失败'
            throw err
        } finally {
            loginLoading.value = false
        }
    }

    /**
     * Handle a successful login/register response.
     */
    function handleLoginResponse(response: { access_token: string; user: UserInfo }): void {
        token.value = response.access_token
        user.value = response.user
        loggedIn.value = true
        localStorage.setItem(TOKEN_KEY, response.access_token)
        setAuthToken(response.access_token)
        fetchQuota()
    }

    /**
     * Logout and clear all auth state.
     */
    function logout(): void {
        clearAuth()
    }

    /**
     * Clear auth state.
     */
    function clearAuth(): void {
        loggedIn.value = false
        user.value = null
        token.value = null
        quota.value = null
        localStorage.removeItem(TOKEN_KEY)
        setAuthToken(null)
    }

    /**
     * Try to restore auth from localStorage token on app load.
     */
    async function tryRestoreAuth(): Promise<void> {
        const stored = localStorage.getItem(TOKEN_KEY)
        if (!stored) return

        setAuthToken(stored)
        try {
            const userInfo = await getMe()
            loggedIn.value = true
            user.value = userInfo
            token.value = stored
            fetchQuota()
        } catch {
            clearAuth()
        }
    }

    return {
        authEnabled,
        turnstileSiteKey,
        loggedIn,
        user,
        token,
        loading,
        loginLoading,
        loginError,
        quota,
        isAdmin,
        quotaRemaining,
        quotaUsed,
        quotaTotal,
        quotaUnlimited,
        checkAuthStatus,
        fetchQuota,
        register,
        login,
        logout,
        tryRestoreAuth,
        clearAuth,
    }
})
