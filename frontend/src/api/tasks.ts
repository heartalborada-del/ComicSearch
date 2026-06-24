/**
 * Task API module.
 * Wraps task submission, status query, listing, and cancellation endpoints.
 */
import { getJson, postJson } from './client'
import type {
    ListTasksParams,
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
 * List tasks with optional filtering.
 * @param params - Optional limit and status filter
 * @returns Array of task records
 */
export function listTasks(params?: ListTasksParams): Promise<TaskRecord[]> {
    const searchParams = new URLSearchParams()
    if (params?.limit !== undefined) {
        searchParams.set('limit', String(params.limit))
    }
    if (params?.status) {
        searchParams.set('status', params.status)
    }
    const query = searchParams.toString()
    return getJson<TaskRecord[]>(`/tasks${query ? `?${query}` : ''}`)
}

/**
 * Cancel a pending or running task.
 * @param taskId - The task ID to cancel
 * @returns Updated task record
 */
export function cancelTask(taskId: string): Promise<TaskRecord> {
    return postJson<TaskRecord>(`/tasks/${taskId}/cancel`, {})
}
