/**
 * Search store — manages search state, parameters, and history.
 * After a successful search, automatically fetches pack metadata (title, keywords)
 * for all candidate manga via the /info endpoint ("联动查询").
 */
import { defineStore } from 'pinia'
import { ref, shallowRef } from 'vue'
import { searchImage } from '@/api/search'
import { getPackInfo } from '@/api/info'
import { ApiError } from '@/api/client'
import type { SearchParams, SearchResponse } from '@/types/search'
import type { PackInfo } from '@/types/info'

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

    /** Cache of pack info keyed by pack_id, populated after each search. */
    const packInfoMap = ref<Record<number, PackInfo>>({})

    /** Whether pack info is still being fetched for the current result set. */
    const packInfoLoading = ref(false)

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
     * After search completes, automatically fetches pack metadata for all candidates.
     */
    async function executeSearch(image: File): Promise<void> {
        loading.value = true
        error.value = null
        packInfoMap.value = {}
        packInfoLoading.value = false

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

            // --- Linked query: fetch pack info for all candidates ---
            const allPackIds = new Set<number>()
            for (const candidate of response.candidate_manga) {
                allPackIds.add(candidate.pack_id)
            }
            if (allPackIds.size > 0) {
                packInfoLoading.value = true
                const results = await Promise.allSettled(
                    [...allPackIds].map((id) => getPackInfo(id)),
                )
                const newMap: Record<number, PackInfo> = {}
                const ids = [...allPackIds]
                for (let i = 0; i < ids.length; i++) {
                    const settled = results[i]
                    if (settled.status === 'fulfilled') {
                        newMap[ids[i]] = settled.value
                    }
                }
                packInfoMap.value = newMap
                packInfoLoading.value = false
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
        packInfoMap.value = {}
        packInfoLoading.value = false
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
        packInfoMap,
        packInfoLoading,
        executeSearch,
        clearResult,
        clearHistory,
    }
})
