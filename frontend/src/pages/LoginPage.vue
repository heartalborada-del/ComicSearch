<script setup lang="ts">
/**
 * Login/Register page — unified auth page with tabbed UI.
 * Requires Turnstile verification when configured on the server.
 */
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import TurnstileWidget from '@/components/TurnstileWidget.vue'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

const tab = ref<'login' | 'register'>('login')

const username = ref('')
const password = ref('')
const turnstileToken = ref<string | null>(null)
const turnstileResetKey = ref(0)
const submitting = computed(() => authStore.loginLoading)
const error = computed(() => authStore.loginError)
const formError = ref<string | null>(null)

const siteKey = computed(() => authStore.turnstileSiteKey)

const canSubmit = computed(() => {
  if (!username.value.trim() || !password.value) return false
  if (siteKey.value && !turnstileToken.value) return false
  return !submitting.value
})

onMounted(() => {
  // Redirect if already logged in
  if (authStore.loggedIn) {
    const redirect = (route.query.redirect as string) || '/'
    router.replace(redirect)
  }

  // Check auth status to get site key
  if (!authStore.authEnabled) {
    authStore.checkAuthStatus()
  }
})

// Clear errors and reset Turnstile when switching tabs
watch(tab, () => {
  formError.value = null
  authStore.loginError = null
  turnstileToken.value = null
  turnstileResetKey.value++
})

function onTurnstileVerify(token: string): void {
  turnstileToken.value = token
  formError.value = null
}

function onTurnstileError(_msg: string): void {
  turnstileToken.value = null
}

async function handleSubmit(): Promise<void> {
  formError.value = null

  if (!username.value.trim()) {
    formError.value = '请输入用户名'
    return
  }
  if (!password.value) {
    formError.value = '请输入密码'
    return
  }
  if (siteKey.value && !turnstileToken.value) {
    formError.value = '请完成验证码'
    return
  }

  try {
    if (tab.value === 'login') {
      await authStore.login(username.value.trim(), password.value, turnstileToken.value ?? undefined)
    } else {
      await authStore.register(username.value.trim(), password.value, turnstileToken.value ?? undefined)
    }

    // Navigate to redirect or home
    const redirect = (route.query.redirect as string) || '/'
    router.replace(redirect)
  } catch {
    // Reset Turnstile on error
    turnstileToken.value = null
    turnstileResetKey.value++
  }
}

</script>

<template>
  <div class="auth-page d-flex align-center justify-center" style="min-height: calc(100vh - 128px)">
    <v-card
      variant="tonal"
      class="bg-surface-container pa-6"
      rounded="xl"
      max-width="420"
      width="100%"
    >
      <!-- Header -->
      <div class="text-center mb-6">
        <v-icon size="48" color="primary" class="mb-2">mdi-shield-account</v-icon>
        <div class="text-h5 font-weight-medium">ComicSearch</div>
        <div class="text-body-2 text-on-surface-variant">登录以使用完整功能</div>
      </div>

      <!-- Tab Switch -->
      <v-tabs v-model="tab" grow class="mb-6" color="primary">
        <v-tab value="login" rounded="lg">登录</v-tab>
        <v-tab value="register" rounded="lg">注册</v-tab>
      </v-tabs>

      <!-- Form -->
      <v-form @submit.prevent="handleSubmit">
        <v-text-field
          v-model="username"
          label="用户名"
          prepend-inner-icon="mdi-account"
          :rules="[v => !!v || '请输入用户名']"
          autocomplete="username"
          class="mb-3"
        />

        <v-text-field
          v-model="password"
          label="密码"
          type="password"
          prepend-inner-icon="mdi-lock"
          :rules="[v => !!v || '请输入密码', v => (v?.length ?? 0) >= 6 || '密码至少6位']"
          autocomplete="current-password"
          class="mb-4"
        />

        <!-- Turnstile -->
        <div v-if="siteKey" class="mb-4">
          <TurnstileWidget
            :site-key="siteKey"
            :reset-key="turnstileResetKey"
            @verify="onTurnstileVerify"
            @error="onTurnstileError"
          />
        </div>

        <!-- Form Error -->
        <v-alert
          v-if="formError"
          type="error"
          variant="tonal"
          density="compact"
          rounded="lg"
          class="mb-4"
          closable
          @click:close="formError = null"
        >
          {{ formError }}
        </v-alert>

        <!-- Server Error -->
        <v-alert
          v-if="error"
          type="error"
          variant="tonal"
          density="compact"
          rounded="lg"
          class="mb-4"
          closable
          @click:close="authStore.loginError = null"
        >
          {{ error }}
        </v-alert>

        <!-- Submit -->
        <v-btn
          color="primary"
          block
          size="large"
          rounded="lg"
          :loading="submitting"
          :disabled="!canSubmit"
          @click="handleSubmit"
        >
          {{ tab === 'login' ? '登录' : '注册' }}
        </v-btn>
      </v-form>
    </v-card>
  </div>
</template>
