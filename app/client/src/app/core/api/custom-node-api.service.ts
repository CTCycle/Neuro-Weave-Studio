// Copyright © 2026 Thomas Virdis
// Licensed under the MIT License.

import { HttpClient, HttpErrorResponse, HttpHeaders } from '@angular/common/http'
import { Injectable, inject } from '@angular/core'
import { firstValueFrom } from 'rxjs'
import type {
  CustomNodeCatalogResponse,
  CustomNodeDefinition,
  CustomNodeDefinitionMutation,
} from '../../domain/custom-node.types'
import { ApiError, apiErrorMessage } from './api-error'

const API_BASE = '/api/nodes/custom'

@Injectable({ providedIn: 'root' })
export class CustomNodeApiService {
  private readonly http = inject(HttpClient)
  private readonly jsonHeaders = new HttpHeaders({ 'Content-Type': 'application/json' })

  async list(): Promise<CustomNodeDefinition[]> {
    const response = await this.request<CustomNodeCatalogResponse>('')
    return response.definitions
  }

  create(payload: CustomNodeDefinitionMutation): Promise<CustomNodeDefinition> {
    return this.request('', 'POST', payload)
  }

  update(id: string, payload: CustomNodeDefinitionMutation): Promise<CustomNodeDefinition> {
    return this.request(`/${encodeURIComponent(id)}`, 'PUT', payload)
  }

  delete(id: string): Promise<void> {
    return this.request(`/${encodeURIComponent(id)}`, 'DELETE')
  }

  private async request<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
    try {
      return await firstValueFrom(this.http.request<T>(method, `${API_BASE}${path}`, {
        body,
        headers: this.jsonHeaders,
        timeout: 10000,
      }))
    } catch (error) {
      throw this.toApiError(error)
    }
  }

  private toApiError(error: unknown): ApiError {
    if (error instanceof ApiError) return error
    if (error instanceof HttpErrorResponse) {
      const payload = error.error as { detail?: unknown } | unknown
      const detail = typeof payload === 'object' && payload !== null && 'detail' in payload
        ? (payload as { detail?: unknown }).detail
        : payload
      return new ApiError(
        error.status,
        apiErrorMessage(detail, error.statusText || 'Request failed.'),
        detail,
      )
    }
    return error instanceof Error ? new ApiError(0, error.message, error) : new ApiError(0, String(error), error)
  }
}
