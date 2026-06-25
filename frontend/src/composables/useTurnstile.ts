/**
 * Cloudflare Turnstile composable.
 * Loads the Turnstile script and provides a callback-based token refresh API.
 * Each call to useTurnstile() creates an independent widget instance.
 */
import { onUnmounted, ref, type Ref } from 'vue'

const TURNSTILE_SCRIPT_URL = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit'

let scriptLoadPromise: Promise<void> | null = null

/** Public API surface of a Turnstile widget instance. */
export interface TurnstileInstance {
    token: Ref<string | null>
    error: Ref<string | null>
    expired: Ref<boolean>
    render: (container: HTMLElement) => void
    reset: () => void
    destroy: () => void
}

/**
 * Load the Cloudflare Turnstile script if not already loaded.
 */
export function loadTurnstileScript(): Promise<void> {
    if (scriptLoadPromise) return scriptLoadPromise

    scriptLoadPromise = new Promise((resolve, reject) => {
        if (typeof window === 'undefined') {
            reject(new Error('window not available'))
            return
        }

        if (window.turnstile) {
            resolve()
            return
        }

        const script = document.createElement('script')
        script.src = TURNSTILE_SCRIPT_URL
        script.async = true
        script.defer = true
        script.onload = () => resolve()
        script.onerror = () => reject(new Error('failed to load turnstile script'))
        document.head.appendChild(script)
    })

    return scriptLoadPromise
}

/**
 * Composable for managing a single Turnstile widget instance.
 */
export function useTurnstile(siteKey: string): TurnstileInstance {
    const token = ref<string | null>(null)
    const error = ref<string | null>(null)
    const expired = ref(false)
    let widgetId: string | null = null

    /**
     * Callback invoked by Turnstile on successful verification.
     */
    function onVerify(turnstileToken: string): void {
        token.value = turnstileToken
        error.value = null
        expired.value = false
    }

    /**
     * Callback invoked when Turnstile token expires.
     */
    function onExpire(): void {
        token.value = null
        expired.value = true
    }

    /**
     * Callback invoked when Turnstile encounters an error.
     */
    function onError(err?: string | Error): void {
        error.value = typeof err === 'string' ? err : err?.message ?? 'turnstile error'
        token.value = null
    }

    /**
     * Render the Turnstile widget in a container element.
     */
    function render(container: HTMLElement): void {
        if (!window.turnstile) return

        if (widgetId) {
            window.turnstile.remove(widgetId)
        }

        widgetId = window.turnstile.render(container, {
            sitekey: siteKey,
            callback: onVerify,
            'expired-callback': onExpire,
            'error-callback': onError,
            theme: 'auto',
            size: 'normal',
        })
    }

    /**
     * Reset the Turnstile widget to get a fresh challenge.
     */
    function reset(): void {
        if (widgetId && window.turnstile) {
            window.turnstile.reset(widgetId)
        }
        token.value = null
        error.value = null
        expired.value = false
    }

    /**
     * Cleanup the widget on unmount.
     */
    function destroy(): void {
        if (widgetId && window.turnstile) {
            window.turnstile.remove(widgetId)
            widgetId = null
        }
        token.value = null
        error.value = null
    }

    onUnmounted(() => {
        destroy()
    })

    return {
        token,
        error,
        expired,
        render,
        reset,
        destroy,
    }
}

/**
 * Type declarations for the Cloudflare Turnstile global.
 */
declare global {
    interface Window {
        turnstile?: {
            render: (
                container: HTMLElement,
                options: Record<string, unknown>,
            ) => string
            remove: (widgetId: string) => void
            reset: (widgetId: string) => void
        }
    }
}
