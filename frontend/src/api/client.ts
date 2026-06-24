/**
 * Unified fetch-based HTTP client.
 * Provides JSON request/response handling, multipart upload support,
 * error extraction from FastAPI HTTPException, and AbortController timeout.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

/** Custom error class for API errors with status code and detail message. */
export class ApiError extends Error {
    constructor(
        message: string,
        public readonly status: number,
        public readonly detail: string,
    ) {
        super(message)
        this.name = 'ApiError'
    }
}

/** Options for the fetch wrapper. */
interface RequestOptions {
    method?: 'GET' | 'POST' | 'PUT' | 'DELETE'
    body?: BodyInit | null
    headers?: Record<string, string>
    signal?: AbortSignal
    timeoutMs?: number
}

/** Default timeout for API requests (30 seconds). */
const DEFAULT_TIMEOUT_MS = 30_000

/**
 * Create an AbortSignal that combines a timeout with an optional external signal.
 */
function createTimeoutSignal(timeoutMs: number, externalSignal?: AbortSignal): AbortSignal {
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeoutMs)

    if (externalSignal) {
        if (externalSignal.aborted) {
            controller.abort()
        } else {
            externalSignal.addEventListener('abort', () => controller.abort(), { once: true })
        }
    }

    // Clear timer when signal aborts
    controller.signal.addEventListener('abort', () => clearTimeout(timer), { once: true })

    return controller.signal
}

/**
 * Build a full URL from the API base URL and a path.
 */
function buildUrl(path: string): string {
    if (path.startsWith('http://') || path.startsWith('https://')) return path
    const base = API_BASE_URL.endsWith('/') ? API_BASE_URL.slice(0, -1) : API_BASE_URL
    const cleanPath = path.startsWith('/') ? path : `/${path}`
    return `${base}${cleanPath}`
}

/**
 * Extract error detail from a FastAPI error response.
 * FastAPI returns { detail: string } for HTTPException.
 */
async function extractError(response: Response): Promise<ApiError> {
    let detail = `HTTP ${response.status}`
    let message = response.statusText || `Request failed with status ${response.status}`

    try {
        const contentType = response.headers.get('content-type') || ''
        if (contentType.includes('application/json')) {
            const body = await response.json()
            if (typeof body === 'object' && body !== null) {
                const rawDetail = (body as Record<string, unknown>).detail
                if (typeof rawDetail === 'string') {
                    detail = rawDetail
                    message = rawDetail
                }
            }
        } else {
            const text = await response.text()
            if (text) {
                detail = text
                message = text
            }
        }
    } catch {
        // Ignore parsing errors
    }

    return new ApiError(message, response.status, detail)
}

/**
 * Core fetch wrapper with timeout, error handling, and JSON parsing.
 */
async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
    const {
        method = 'GET',
        body = null,
        headers = {},
        signal,
        timeoutMs = DEFAULT_TIMEOUT_MS,
    } = options

    const finalSignal = createTimeoutSignal(timeoutMs, signal)

    const response = await fetch(buildUrl(path), {
        method,
        body,
        headers,
        signal: finalSignal,
    })

    if (!response.ok) {
        throw await extractError(response)
    }

    // Handle empty responses (e.g., 204 No Content)
    if (response.status === 204) {
        return undefined as T
    }

    const contentType = response.headers.get('content-type') || ''
    if (contentType.includes('application/json')) {
        return (await response.json()) as T
    }

    return (await response.text()) as unknown as T
}

/** Perform a GET request expecting JSON. */
export function getJson<T>(path: string, options?: Omit<RequestOptions, 'method' | 'body'>): Promise<T> {
    return request<T>(path, { ...options, method: 'GET' })
}

/** Perform a POST request with a JSON body. */
export function postJson<T>(
    path: string,
    data: unknown,
    options?: Omit<RequestOptions, 'method' | 'body'>,
): Promise<T> {
    return request<T>(path, {
        ...options,
        method: 'POST',
        body: JSON.stringify(data),
        headers: { 'Content-Type': 'application/json', ...options?.headers },
    })
}

/** Perform a POST request with a FormData body (multipart/form-data). */
export function postForm<T>(
    path: string,
    formData: FormData,
    options?: Omit<RequestOptions, 'method' | 'body' | 'headers'>,
): Promise<T> {
    // Do not set Content-Type header — the browser sets it with the correct boundary
    return request<T>(path, {
        ...options,
        method: 'POST',
        body: formData,
    })
}
