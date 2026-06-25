<script setup lang="ts">
/**
 * Turnstile widget component — renders a Cloudflare Turnstile challenge.
 * Emits a verified token on successful completion.
 */
import { nextTick, onMounted, ref, watch } from 'vue'
import { loadTurnstileScript, useTurnstile } from '@/composables/useTurnstile'

const emit = defineEmits<{
  verify: [token: string]
  error: [message: string]
}>()

const props = withDefaults(
  defineProps<{
    siteKey: string
    resetKey?: number
  }>(),
  {
    resetKey: 0,
  },
)

const containerRef = ref<HTMLElement | null>(null)
const loading = ref(true)
const loadError = ref<string | null>(null)

const turnstile = props.siteKey
  ? useTurnstile(props.siteKey)
  : null

onMounted(async () => {
  if (!props.siteKey) {
    loading.value = false
    return
  }

  try {
    await loadTurnstileScript()
    await nextTick()
    if (containerRef.value && turnstile) {
      turnstile.render(containerRef.value)
    }
  } catch (err) {
    loadError.value = '验证码加载失败'
    emit('error', '验证码加载失败')
  } finally {
    loading.value = false
  }
})

// Watch token changes and emit verify
watch(
  () => turnstile?.token.value,
  (newToken) => {
    if (newToken) {
      emit('verify', newToken)
    }
  },
)

// Watch for reset key changes
watch(
  () => props.resetKey,
  () => {
    turnstile?.reset()
  },
)
</script>

<template>
  <div class="turnstile-widget">
    <!-- Loading state -->
    <div
      v-if="loading"
      class="d-flex align-center justify-center pa-4"
      style="min-height: 65px"
    >
      <v-progress-circular
        indeterminate
        size="20"
        width="2"
        color="primary"
        class="mr-2"
      />
      <span class="text-caption text-on-surface-variant">加载验证码…</span>
    </div>

    <!-- Error state -->
    <v-alert
      v-else-if="loadError"
      type="warning"
      variant="tonal"
      density="compact"
      rounded="lg"
      closable
    >
      {{ loadError }}
    </v-alert>

    <!-- Turnstile container -->
    <div
      v-else-if="siteKey"
      ref="containerRef"
      class="turnstile-container d-flex justify-center"
    />

    <!-- No site key configured -->
    <div v-else class="text-caption text-on-surface-variant text-center pa-2">
      验证码未配置
    </div>
  </div>
</template>

<style scoped lang="scss">
.turnstile-container {
  min-height: 65px;
}
</style>
