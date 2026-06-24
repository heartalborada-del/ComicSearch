/**
 * Search store — manages search state, parameters, and history.
 */
import { defineStore } from 'pinia'
import { ref, shallowRef } from 'vue'
import { searchImage } from '@/api/search'
import { ApiError } from '@/api/client'
import type { SearchParams, SearchResponse } from '@/types/search'

/** Maximum number of search history entries to keep. */
const MAX_HISTORY = 10

/** A search history entry (stores thumbnail, not original image). */
export interface SearchHistoryEntry {
    id: string
    timestamp: number
    thumbnailUrl: string
    confidence: string
    bestPackId: number | null
    bestScore: number | null
}

export const useSearchStore = defineStore('search', () => {
    const loading = ref(false)
    const error = ref<string | null>(null)
    const result = shallowRef<SearchResponse | null>(null)
    const history = ref<SearchHistoryEntry[]>([])

    /** Default search parameters. */
    const params = ref<SearchParams>({
        robust_partial: true,
        include_corners: true,
        include_contrast: false,
        per_view_limit: 80,
        top_k_manga: 10,
        keyword_ids: [],
    })

    /**
     * Execute a search with the given image file.
     */
    async function executeSearch(image: File): Promise<void> {
        loading.value = true
        error.value = null

        try {
            const response = await searchImage(image, params.value)
            result.value = response

            // Add to history
            const thumbnailUrl = URL.createObjectURL(image)
            const entry: SearchHistoryEntry = {
                id: crypto.randomUUID(),
                timestamp: Date.now(),
                thumbnailUrl,
                confidence: response.confidence,
                bestPackId: response.best_manga?.pack_id ?? null,
                bestScore: response.best_manga?.score ?? null,
            }
            history.value.unshift(entry)
            if (history.value.length > MAX_HISTORY) {
                const removed = history.value.pop()
                if (removed) URL.revokeObjectURL(removed.thumbnailUrl)
            }
        } catch (err) {
            if (err instanceof ApiError) {
                error.value = err.detail || err.message
            } else if (err instanceof Error) {
                error.value = err.message
            } else {
                error.value = '搜索失败，请重试'
            }
            result.value = null
        } finally {
            loading.value = false
        }
    }

    /** Clear current search results. */
    function clearResult(): void {
        result.value = null
        error.value = null
    }

    /** Clear search history and revoke object URLs. */
    function clearHistory(): void {
        for (const entry of history.value) {
            URL.revokeObjectURL(entry.thumbnailUrl)
        }
        history.value = []
    }

    return {
        loading,
        error,
        result,
        history,
        params,
        executeSearch,
        clearResult,
        clearHistory,
    }
})
