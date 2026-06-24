<script setup lang="ts">
/**
 * Detail page — displays pack metadata including title, source, keywords,
 * and cover image. Handles loading, error, and not-found states.
 */
import { onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { getPackInfo } from '@/api/info'
import { ApiError } from '@/api/client'
import { coverUrl } from '@/utils/image-url'
import type { PackInfo } from '@/types/info'
import KeywordChip from '@/components/KeywordChip.vue'

const route = useRoute()

const packInfo = ref<PackInfo | null>(null)
const loading = ref(true)
const error = ref<string | null>(null)
const notFound = ref(false)

const packId = ref<number>(0)

const coverSrc = ref<string>('')

async function loadInfo(id: number): Promise<void> {
  loading.value = true
  error.value = null
  notFound.value = false
  packInfo.value = null

  try {
    packInfo.value = await getPackInfo(id)
    coverSrc.value = coverUrl(id)
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      notFound.value = true
    } else {
      error.value = err instanceof ApiError ? err.detail : '获取详情失败'
    }
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  const id = parseInt(route.params.id as string, 10)
  if (!isNaN(id) && id > 0) {
    packId.value = id
    loadInfo(id)
  } else {
    error.value = '无效的 Pack ID'
    loading.value = false
  }
})

watch(
  () => route.params.id,
  (newId) => {
    const id = parseInt(newId as string, 10)
    if (!isNaN(id) && id > 0 && id !== packId.value) {
      packId.value = id
      loadInfo(id)
    }
  },
)
</script>

<template>
  <div class="detail-page">
    <!-- Loading -->
    <div v-if="loading" class="d-flex justify-center py-12">
      <v-progress-circular indeterminate color="primary" size="48" width="4" />
    </div>

    <!-- Not Found -->
    <div v-else-if="notFound" class="d-flex flex-column align-center py-12 text-center">
      <v-icon size="80" color="error" class="mb-4">mdi-book-off-outline</v-icon>
      <div class="text-h6 text-on-surface-variant mb-1">未找到该漫画</div>
      <div class="text-body-2 text-on-surface-variant mb-4">
        Pack ID: {{ packId }} 不存在
      </div>
      <v-btn variant="tonal" color="primary" to="/" prepend-icon="mdi-arrow-left">
        返回搜索
      </v-btn>
    </div>

    <!-- Error -->
    <v-alert v-else-if="error" type="error" variant="tonal" rounded="lg">
      {{ error }}
    </v-alert>

    <!-- Detail Content -->
    <div v-else-if="packInfo" class="detail-content">
      <v-row>
        <!-- Cover Image -->
        <v-col cols="12" sm="4" md="3">
          <v-card variant="tonal" class="bg-surface-container" rounded="xl" overflow="hidden">
            <v-img
              :src="coverSrc"
              aspect-ratio="0.7"
              cover
              class="bg-surface-container-high"
            >
              <template #placeholder>
                <div class="d-flex align-center justify-center fill-height">
                  <v-icon size="64" color="outline">mdi-book-outline</v-icon>
                </div>
              </template>
              <template #error>
                <div class="d-flex align-center justify-center fill-height">
                  <v-icon size="64" color="outline">mdi-book-off-outline</v-icon>
                </div>
              </template>
            </v-img>
          </v-card>
        </v-col>

        <!-- Metadata -->
        <v-col cols="12" sm="8" md="9">
          <div class="text-overline text-on-surface-variant mb-1">
            Pack #{{ packInfo.pack_id }}
          </div>
          <h1 class="text-h5 font-weight-medium mb-3">
            {{ packInfo.title || '未命名漫画' }}
          </h1>

          <!-- Source Link -->
          <div v-if="packInfo.source" class="mb-4">
            <div class="text-subtitle-2 text-on-surface-variant mb-1">来源</div>
            <v-btn
              :href="packInfo.source"
              target="_blank"
              rel="noopener noreferrer"
              variant="tonal"
              color="primary"
              size="small"
              prepend-icon="mdi-open-in-new"
            >
              {{ packInfo.source }}
            </v-btn>
          </div>

          <!-- Keywords -->
          <div v-if="packInfo.keywords.length > 0" class="mb-4">
            <div class="text-subtitle-2 text-on-surface-variant mb-2">关键词</div>
            <div class="d-flex flex-wrap ga-2">
              <KeywordChip
                v-for="keyword in packInfo.keywords"
                :key="keyword.id"
                :keyword="keyword"
              />
            </div>
          </div>

          <!-- Keyword IDs (raw) -->
          <div v-if="packInfo.keyword_ids.length > 0">
            <div class="text-subtitle-2 text-on-surface-variant mb-1">关键词 ID</div>
            <div class="text-body-2 text-on-surface-variant">
              {{ packInfo.keyword_ids.join(', ') }}
            </div>
          </div>
        </v-col>
      </v-row>
    </div>
  </div>
</template>
