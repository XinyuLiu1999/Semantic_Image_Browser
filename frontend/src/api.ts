// API client for ImageNet21K backend

import type { SearchResponse, TaxonomyNodeDetails, BrowseResponse, Suggestion, SystemStats } from './types';

const API_BASE = '/api';

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

export const api = {
  /**
   * Perform semantic search
   */
  async search(
    query: string,
    page: number = 1,
    limit: number = 20,
    includeDescendants: boolean = true
  ): Promise<SearchResponse> {
    const params = new URLSearchParams({
      q: query,
      page: String(page),
      limit: String(limit),
      include_descendants: String(includeDescendants)
    });
    return fetchJson<SearchResponse>(`${API_BASE}/search?${params}`);
  },

  /**
   * Get autocomplete suggestions
   */
  async getSuggestions(query: string, limit: number = 10): Promise<Suggestion[]> {
    if (query.length < 2) return [];
    const params = new URLSearchParams({
      q: query,
      limit: String(limit)
    });
    return fetchJson<Suggestion[]>(`${API_BASE}/suggestions?${params}`);
  },

  /**
   * Get taxonomy node details
   */
  async getTaxonomyNode(nodeId: string): Promise<TaxonomyNodeDetails> {
    return fetchJson<TaxonomyNodeDetails>(`${API_BASE}/taxonomy/${nodeId}`);
  },

  /**
   * Browse images under a taxonomy node
   */
  async browseNode(
    nodeId: string,
    page: number = 1,
    limit: number = 20,
    includeDescendants: boolean = true
  ): Promise<BrowseResponse> {
    const params = new URLSearchParams({
      page: String(page),
      limit: String(limit),
      include_descendants: String(includeDescendants)
    });
    return fetchJson<BrowseResponse>(`${API_BASE}/browse/${nodeId}?${params}`);
  },

  /**
   * Get image file URL
   */
  getImageUrl(imageId: string): string {
    return `${API_BASE}/images/${imageId}/file`;
  },

  /**
   * Get system statistics
   */
  async getStats(): Promise<SystemStats> {
    return fetchJson<SystemStats>(`${API_BASE}/stats`);
  },

  /**
   * Health check
   */
  async healthCheck(): Promise<{ status: string; taxonomy_loaded: boolean; images_loaded: boolean }> {
    return fetchJson(`${API_BASE}/health`);
  }
};
