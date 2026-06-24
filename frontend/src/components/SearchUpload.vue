<script setup lang="ts">
/**
 * Search upload — image upload component with drag-and-drop,
 * file picker, preview, and client-side validation.
 * Validates file type (jpeg/png/webp) and size (max 10MB).
 */
import { computed, ref } from 'vue'

const MAX_SIZE_BYTES = 10 * 1024 * 1024
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp']

const emit = defineEmits<{
  'update:file': [file: File | null]
}>()

const props = defineProps<{
  file: File | null
}>()

const isDragging = ref(false)
const error = ref<string | null>(null)
const previewUrl = ref<string | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)

const hasFile = computed(() => props.file !== null)

function triggerFileInput(): void {
  fileInput.value?.click()
}

/**
 * Validate a file's type and size.
 */
function validateFile(file: File): string | null {
  if (!ALLOWED_TYPES.includes(file.type)) {
    return `不支持的图片格式: ${file.type}，仅支持 JPEG、PNG、WebP`
  }
  if (file.size > MAX_SIZE_BYTES) {
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1)
    return `图片大小 ${sizeMB}MB 超过限制（最大 10MB）`
  }
  return null
}

/**
 * Process a selected file: validate, create preview, emit update.
 */
function processFile(file: File): void {
  const validationError = validateFile(file)
  if (validationError) {
    error.value = validationError
    emit('update:file', null)
    return
  }

  error.value = null
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
  }
  previewUrl.value = URL.createObjectURL(file)
  emit('update:file', file)
}

/**
 * Handle file input change.
 */
function onFileChange(event: Event): void {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    processFile(file)
  }
}

/**
 * Handle drag over.
 */
function onDragOver(event: DragEvent): void {
  event.preventDefault()
  isDragging.value = true
}

/**
 * Handle drag leave.
 */
function onDragLeave(): void {
  isDragging.value = false
}

/**
 * Handle drop.
 */
function onDrop(event: DragEvent): void {
  event.preventDefault()
  isDragging.value = false
  const file = event.dataTransfer?.files?.[0]
  if (file) {
    processFile(file)
  }
}

/**
 * Clear the selected file.
 */
function clearFile(): void {
  if (previewUrl.value) {
    URL.revokeObjectURL(previewUrl.value)
    previewUrl.value = null
  }
  error.value = null
  emit('update:file', null)
}
</script>

<template>
  <div class="search-upload">
    <!-- Drop zone / Preview -->
    <div
      v-if="!hasFile"
      class="drop-zone"
      :class="{ 'drop-zone--active': isDragging }"
      @dragover="onDragOver"
      @dragleave="onDragLeave"
      @drop="onDrop"
      @click="triggerFileInput"
    >
      <v-icon size="64" color="primary" class="mb-3">mdi-cloud-upload-outline</v-icon>
      <div class="text-body-1 text-on-surface font-weight-medium mb-1">
        拖拽图片到此处或点击选择
      </div>
      <div class="text-caption text-on-surface-variant">
        支持 JPEG、PNG、WebP，最大 10MB
      </div>
      <input
        ref="fileInput"
        type="file"
        class="d-none"
        accept="image/jpeg,image/png,image/webp"
        @change="onFileChange"
      />
    </div>

    <!-- Preview with file info -->
    <v-card v-else variant="tonal" class="bg-surface-container pa-3" rounded="lg">
      <div class="d-flex align-center ga-3">
        <v-img
          :src="previewUrl || ''"
          max-width="80"
          max-height="80"
          cover
          rounded="md"
          class="flex-shrink-0"
        />
        <div class="flex-grow-1 overflow-hidden">
          <div class="text-body-2 font-weight-medium text-truncate">{{ file?.name }}</div>
          <div class="text-caption text-on-surface-variant">
            {{ file ? (file.size / 1024).toFixed(0) : 0 }} KB · {{ file?.type }}
          </div>
        </div>
        <v-btn
          icon="mdi-close"
          variant="text"
          size="small"
          @click="clearFile"
        />
      </div>
    </v-card>

    <!-- Error message -->
    <v-alert
      v-if="error"
      type="error"
      variant="tonal"
      density="compact"
      class="mt-2"
      closable
      @click:close="error = null"
    >
      {{ error }}
    </v-alert>
  </div>
</template>

<style scoped lang="scss">
.drop-zone {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  border: 2px dashed rgb(var(--v-theme-outline-variant));
  border-radius: 1rem;
  cursor: pointer;
  transition: border-color 0.2s ease, background-color 0.2s ease;
  padding: 2rem;
  text-align: center;
}

.drop-zone:hover {
  border-color: rgb(var(--v-theme-primary));
  background: rgb(var(--v-theme-primary-container) / 0.1);
}

.drop-zone--active {
  border-color: rgb(var(--v-theme-primary));
  background: rgb(var(--v-theme-primary-container) / 0.2);
}
</style>
