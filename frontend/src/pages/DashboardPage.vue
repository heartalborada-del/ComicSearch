<script setup lang="ts">
/**
 * Dashboard page — displays global stats: total packs and total tags.
 */
import { onMounted, ref } from 'vue'
import { getStats } from '@/api/info'
import { ApiError } from '@/api/client'
import type { StatsResponse } from '@/types/info'

const stats = ref<StatsResponse | null>(null)
const loading = ref(true)
const error = ref<string | null>(null)

async function loadStats(): Promise<void> {
  loading.value = true
  error.value = null
  try {
    stats.value = await getStats()
  } catch (err) {
    error.value = err instanceof ApiError ? err.detail : '获取统计数据失败'
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  loadStats()
})
</script>

<template>
  <div class="dashboard-page px-4 px-sm-6 px-md-8">
    <h1 class="text-h4 mb-6">数据总览</h1>

    <!-- Loading -->
    <div v-if="loading" class="d-flex justify-center py-12">
      <v-progress-circular indeterminate color="primary" size="48" width="4" />
    </div>

    <!-- Error -->
    <v-alert
      v-else-if="error"
      type="error"
      variant="tonal"
      class="mb-4"
      closable
    >
      {{ error }}
    </v-alert>

    <!-- Stats Cards -->
    <v-row v-else-if="stats">
      <v-col cols="12" sm="6" md="4">
        <v-card variant="tonal" class="bg-surface-container" rounded="xl">
          <v-card-item>
            <template #prepend>
              <v-avatar color="primary-container" size="56" rounded="lg">
                <v-icon color="on-primary-container" size="28">mdi-book-open-page-variant-outline</v-icon>
              </v-avatar>
            </template>
            <div class="text-label-large text-on-surface-variant">漫画包总数</div>
            <div class="text-h3 font-weight-bold text-on-surface mt-1">
              {{ stats.pack_count.toLocaleString() }}
            </div>
          </v-card-item>
        </v-card>
      </v-col>

      <v-col cols="12" sm="6" md="4">
        <v-card variant="tonal" class="bg-surface-container" rounded="xl">
          <v-card-item>
            <template #prepend>
              <v-avatar color="tertiary-container" size="56" rounded="lg">
                <v-icon color="on-tertiary-container" size="28">mdi-tag-multiple-outline</v-icon>
              </v-avatar>
            </template>
            <div class="text-label-large text-on-surface-variant">标签总数</div>
            <div class="text-h3 font-weight-bold text-on-surface mt-1">
              {{ stats.keyword_count.toLocaleString() }}
            </div>
          </v-card-item>
        </v-card>
      </v-col>
    </v-row>
  </div>
</template>
