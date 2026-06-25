<script setup lang="ts">
/**
 * Import page — submit E-Hentai URLs for import as async tasks.
 * Displays submission results with duplicate detection.
 */
import { computed, onUnmounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useTasksStore } from '@/stores/tasks'

const router = useRouter()
const tasksStore = useTasksStore()

const urlInput = ref('')
const cropFaces = ref(true)

/** Parse textarea input into individual URLs. */
const parsedUrls = computed<string[]>(() => {
  return urlInput.value
    .split(/[\n,，]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
})

const canSubmit = computed(() => parsedUrls.value.length > 0 && !tasksStore.submitting)

async function handleSubmit(): Promise<void> {
  if (parsedUrls.value.length === 0) return

  await tasksStore.submitImportTask({
    urls: parsedUrls.value,
    crop_faces: cropFaces.value,
  })
}

function goToTasks(): void {
  router.push({ name: 'tasks' })
}

onUnmounted(() => {
  tasksStore.stopPolling()
})
</script>

<template>
  <div class="import-page">
    <h1 class="text-h5 font-weight-medium mb-4">E-Hentai 导入</h1>

    <!-- URL Input -->
    <v-card variant="tonal" class="bg-surface-container mb-4 pa-4" rounded="lg">
      <div class="text-subtitle-1 font-weight-medium mb-2">输入漫画 URL</div>
      <v-textarea v-model="urlInput" label="每行一个 URL"
        placeholder="https://e-hentai.org/g/12345/abcdef/&#10;https://e-hentai.org/g/67890/ghijkl/" rows="5" auto-grow
        variant="outlined" :hint="`已输入 ${parsedUrls.length} 个 URL`" persistent-hint />

      <v-switch v-model="cropFaces" label="人脸裁剪" density="compact" class="mt-2" hide-details />

      <div class="d-flex ga-3 mt-4 flex-wrap">
        <v-btn color="primary" rounded="lg" prepend-icon="mdi-cloud-upload" :loading="tasksStore.submitting"
          :disabled="!canSubmit" @click="handleSubmit">
          提交导入
        </v-btn>
        <v-btn variant="tonal" rounded="lg" prepend-icon="mdi-format-list-checks" :disabled="!tasksStore.submitResult"
          @click="goToTasks">
          查看任务
        </v-btn>
      </div>
    </v-card>

    <!-- Submit Error -->
    <v-alert v-if="tasksStore.submitError" type="error" variant="tonal" rounded="lg" class="mb-4" closable>
      {{ tasksStore.submitError }}
    </v-alert>

    <!-- Submit Results -->
    <v-card v-if="tasksStore.submitResult" variant="tonal" class="bg-surface-container pa-4" rounded="lg">
      <div class="text-subtitle-1 font-weight-medium mb-3">提交结果</div>
      <v-list variant="flat" rounded="lg" class="bg-surface-container-high">
        <v-list-item v-for="item in tasksStore.submitResult.items" :key="item.task_id" class="mb-1" rounded="lg">
          <template #prepend>
            <v-icon :color="item.is_duplicate ? 'warning' : 'success'"
              :icon="item.is_duplicate ? 'mdi-content-copy' : 'mdi-check-circle'" />
          </template>
          <v-list-item-title class="text-body-2 text-truncate">
            {{ item.url }}
          </v-list-item-title>
          <v-list-item-subtitle>
            Task: {{ item.task_id.slice(0, 8) }}… · 状态: {{ item.status }}
          </v-list-item-subtitle>
          <template #append>
            <v-chip :color="item.is_duplicate ? 'warning' : 'success'" size="small" variant="tonal">
              {{ item.is_duplicate ? '重复' : '新建' }}
            </v-chip>
          </template>
        </v-list-item>
      </v-list>
    </v-card>
  </div>
</template>
