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
      // Type-aware recommended: turns on checks that need type information
      // (floating promises, misused promises, unsafe any crossings, ...).
      // parserOptions.projectService below wires the TS project service so
      // every .ts/.tsx file — tests included — is linted with real types.
      tseslint.configs.recommendedTypeChecked,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
      parserOptions: {
        // typescript-eslint v8 standard: let the TS project service resolve
        // tsconfig.app.json (src/**) / tsconfig.node.json (vite.config.ts)
        // per file instead of maintaining a manual 'project' glob list.
        projectService: {
          // vitest.config.ts is intentionally not part of any tsconfig
          // project; lint it with the compiler's default options instead.
          allowDefaultProject: ['vitest.config.ts'],
        },
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      // Context modules intentionally colocate their provider and hook.
      'react-refresh/only-export-components': 'off',
    },
  },
  {
    // axios response 'data' is 'any' by design; this module is the single
    // typed boundary where backend payloads enter the app. Every export
    // already declares its return type, so typing ~50 call sites would just
    // duplicate those declarations. Narrow the unsafe-any rules to this
    // file instead of disabling them globally.
    files: ['src/services/api.ts'],
    rules: {
      '@typescript-eslint/no-unsafe-assignment': 'off',
      '@typescript-eslint/no-unsafe-argument': 'off',
      '@typescript-eslint/no-unsafe-member-access': 'off',
      '@typescript-eslint/no-unsafe-return': 'off',
    },
  },
  {
    // Test files reference mocked methods as values (e.g.
    // expect(URL.createObjectURL)) and assert on spies — classic
    // unbound-method false positives in vitest code.
    files: ['**/*.test.ts', '**/*.test.tsx'],
    rules: {
      '@typescript-eslint/unbound-method': 'off',
    },
  },
])
