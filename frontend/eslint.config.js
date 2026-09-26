import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      // recommendedTypeChecked was evaluated and rejected: it surfaces 100+
      // violations across files owned by other workstreams (unsafe any from
      // axios/JSON.parse, no-misused-promises on existing handlers). Type
      // safety is enforced by "strict": true in tsconfig.app/node.json.
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
    },
    rules: {
      // Context modules intentionally colocate their provider and hook.
      'react-refresh/only-export-components': 'off',
    },
  },
])
