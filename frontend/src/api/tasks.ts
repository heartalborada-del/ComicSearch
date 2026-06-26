/**
 * Task API module.
 * Wraps task submission, status query, listing, and cancellation endpoints.
 */
import { getJson, postJson } from './client'
import type {
    ListTasksParams,
    PaginatedTasks,
    SubmitImportRequest,
    SubmitImportResponse,
    TaskRecord,
} from '@/types/task'

/**
 * Submit an E-Hentai import task.
 * @param request - Import request with URL(s) and crop_faces option
 * @returns Submission response with task IDs and statuses
 */
export function submitImport(request: SubmitImportRequest): Promise<SubmitImportResponse> {
    return postJson<SubmitImportResponse>('/ehentai/import/tasks', request)
}

/**
 * Get the status of a single task.
 * @param taskId - The task ID to query
 * @returns Task record with current status
 */
export function getTask(taskId: string): Promise<TaskRecord> {
    return getJson<TaskRecord>(`/tasks/${taskId}`)
}

/**
 * List tasks with optional filtering and pagination.
 * @param params - Optional limit, offset, and status filter
 * @returns Paginated tasks response with items and total count
 */
export function listTasks(params?: ListTasksParams): Promise<PaginatedTasks> {
    const searchParams = new URLSearchParams()
    if (params?.limit !== undefined) {
        searchParams.set('limit', String(params.limit))
    }
    if (params?.offset !== undefined) {
        searchParams.set('offset', String(params.offset))
    }
    if (params?.status) {
        searchParams.set('status', params.status)
    }
    const query = searchParams.toString()
    return getJson<PaginatedTasks>(`/tasks${query ? `?${query}` : ''}`)
}

/**
 * Cancel a pending or running task.
 * @param taskId - The task ID to cancel
 * @returns Updated task record
 */
export function cancelTask(taskId: string): Promise<TaskRecord> {
    return postJson<TaskRecord>(`/tasks/${taskId}/cancel`, {})
}

/**
 * Admin: list tasks pending review.
 */
export function listReviewTasks(): Promise<PaginatedTasks> {
    return getJson<PaginatedTasks>('/tasks/review')
}

/**
 * Admin: approve a pending_review task.
 */
export function approveTask(taskId: string): Promise<TaskRecord> {
    return postJson<TaskRecord>(`/tasks/${taskId}/approve`, {})
}

/**
 * Admin: reject a pending_review task.
 */
export function rejectTask(taskId: string): Promise<TaskRecord> {
    return postJson<TaskRecord>(`/tasks/${taskId}/reject`, {})
}
