import { createReadStream } from 'node:fs'
import { access, stat } from 'node:fs/promises'
import { createServer } from 'node:http'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { request as httpRequest } from 'node:http'

const clientRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const distRoot = path.resolve(clientRoot, '../tests/cache/frontend-dist')
const settings = await readSettings(path.resolve(clientRoot, '../../settings/.env'))
const environment = { ...settings, ...process.env }
const apiHost = environment.FASTAPI_HOST || '127.0.0.1'
const apiPort = Number(environment.FASTAPI_PORT || 8000)
const uiHost = argumentValue('--host') || environment.UI_HOST || '127.0.0.1'
const uiPort = Number(argumentValue('--port') || environment.UI_PORT || 5173)
const mimeTypes = { '.css': 'text/css; charset=utf-8', '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.json': 'application/json; charset=utf-8', '.map': 'application/json; charset=utf-8', '.svg': 'image/svg+xml', '.woff': 'font/woff', '.woff2': 'font/woff2' }

const server = createServer(async (request, response) => {
  const requestUrl = new URL(request.url || '/', `http://${request.headers.host || 'localhost'}`)
  if (requestUrl.pathname === '/health' || requestUrl.pathname === '/api' || requestUrl.pathname.startsWith('/api/')) {
    proxyRequest(request, response, requestUrl.pathname + requestUrl.search)
    return
  }
  const relativePath = decodeURIComponent(requestUrl.pathname).replace(/^\/+/, '')
  const candidate = path.resolve(distRoot, relativePath)
  if (!candidate.startsWith(`${distRoot}${path.sep}`) && candidate !== distRoot) { response.writeHead(400); response.end('Invalid path'); return }
  const filePath = await fileIfPresent(candidate) || path.join(distRoot, 'index.html')
  try {
    const metadata = await stat(filePath)
    response.writeHead(200, { 'Content-Type': mimeTypes[path.extname(filePath)] || 'application/octet-stream', 'Content-Length': metadata.size, 'Cache-Control': filePath.endsWith('index.html') ? 'no-cache' : 'public, max-age=31536000, immutable' })
    createReadStream(filePath).pipe(response)
  } catch {
    response.writeHead(503)
    response.end('Frontend build output is missing. Run npm run build first.')
  }
})

server.listen(uiPort, uiHost, () => console.log(`NeuroWeave preview listening at http://${uiHost}:${uiPort}/`))

function proxyRequest(request, response, requestPath) {
  const upstream = httpRequest({ hostname: apiHost, port: apiPort, path: requestPath, method: request.method, headers: { ...request.headers, host: `${apiHost}:${apiPort}` } }, (upstreamResponse) => {
    response.writeHead(upstreamResponse.statusCode || 502, upstreamResponse.headers)
    upstreamResponse.pipe(response)
  })
  upstream.on('error', (error) => { if (!response.headersSent) response.writeHead(502); response.end(`Backend unavailable: ${error.message}`) })
  request.pipe(upstream)
}

async function fileIfPresent(filePath) { try { await access(filePath); return filePath } catch { return null } }
function argumentValue(name) { const index = process.argv.indexOf(name); return index >= 0 ? process.argv[index + 1] : undefined }
async function readSettings(filePath) {
  try {
    const text = await (await import('node:fs/promises')).readFile(filePath, 'utf8')
    return Object.fromEntries(text.split(/\r?\n/).filter((line) => line.trim() && !line.trim().startsWith('#')).flatMap((line) => { const index = line.indexOf('='); return index > 0 ? [[line.slice(0, index).trim(), line.slice(index + 1).trim().replace(/^['"]|['"]$/g, '')]] : [] }))
  } catch { return {} }
}
