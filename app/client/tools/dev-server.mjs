// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { spawn } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const clientRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const cacheRoot = path.resolve(clientRoot, '../../runtimes/cache')
const settings = await readSettings(path.resolve(clientRoot, '../../settings/.env'))
const environment = { ...settings, ...process.env }
const apiHost = normalizeConnectHost(environment.FASTAPI_HOST || '127.0.0.1')
const apiPort = environment.FASTAPI_PORT || '8000'
const uiHost = argumentValue('--host') || environment.UI_HOST || '127.0.0.1'
const uiPort = argumentValue('--port') || environment.UI_PORT || '5173'
const proxyPath = path.join(cacheRoot, 'angular-proxy.json')

await mkdir(cacheRoot, { recursive: true })
await writeFile(proxyPath, JSON.stringify({
  '/runtime-config.json': { target: `http://${apiHost}:${apiPort}`, changeOrigin: true },
  '/api': { target: `http://${apiHost}:${apiPort}`, changeOrigin: true, ws: true },
  '/health': { target: `http://${apiHost}:${apiPort}`, changeOrigin: true },
}, null, 2))

const cli = path.join(clientRoot, 'node_modules/@angular/cli/bin/ng.js')
const child = spawn(process.execPath, [cli, 'serve', '--configuration', 'development', '--host', uiHost, '--port', uiPort, '--proxy-config', proxyPath], {
  cwd: clientRoot,
  env: environment,
  stdio: 'inherit',
  windowsHide: true,
})

const forwardSignal = (signal) => { if (!child.killed) child.kill(signal) }
process.on('SIGINT', () => forwardSignal('SIGINT'))
process.on('SIGTERM', () => forwardSignal('SIGTERM'))
child.on('exit', (code, signal) => process.exit(code ?? (signal ? 1 : 0)))

async function readSettings(filePath) {
  try {
    const text = await (await import('node:fs/promises')).readFile(filePath, 'utf8')
    return Object.fromEntries(text.split(/\r?\n/).filter((line) => line.trim() && !line.trim().startsWith('#')).flatMap((line) => {
      const index = line.indexOf('=')
      return index > 0 ? [[line.slice(0, index).trim(), line.slice(index + 1).trim().replace(/^['"]|['"]$/g, '')]] : []
    }))
  } catch {
    return {}
  }
}

function argumentValue(name) {
  const index = process.argv.indexOf(name)
  return index >= 0 ? process.argv[index + 1] : undefined
}

function normalizeConnectHost(host) {
  return host === '0.0.0.0' || host === '::' ? '127.0.0.1' : host
}
