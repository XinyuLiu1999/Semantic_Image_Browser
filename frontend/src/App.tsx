import { useState, useEffect, useCallback } from 'react';
import { SearchBar } from './components/SearchBar';
import { TaxonomyBreadcrumb } from './components/TaxonomyBreadcrumb';
import { MasonryGrid } from './components/MasonryGrid';
import { Pagination } from './components/Pagination';
import { StatsBar } from './components/StatsBar';
import { api } from './api';
import type { SearchResponse, SystemStats } from './types';

const ITEMS_PER_PAGE = 24;

function App() {
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentQuery, setCurrentQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [stats, setStats] = useState<SystemStats | null>(null);
  // Track if we're browsing a specific node (vs semantic search)
  const [browsingNodeId, setBrowsingNodeId] = useState<string | null>(null);
  // Preserve original matched nodes from search when browsing
  const [originalMatchedNodes, setOriginalMatchedNodes] = useState<SearchResponse['matched_nodes'] | null>(null);

  // Load system stats on mount
  useEffect(() => {
    api.getStats()
      .then(setStats)
      .catch(console.error);
  }, []);

  const performSearch = useCallback(async (query: string, page: number = 1) => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setCurrentQuery(query);
    setCurrentPage(page);
    setBrowsingNodeId(null);  // Clear browse mode when searching
    setOriginalMatchedNodes(null);  // Clear original matched nodes

    try {
      const result = await api.search(query, page, ITEMS_PER_PAGE);
      setSearchResult(result);
      setOriginalMatchedNodes(result.matched_nodes);  // Save matched nodes
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed');
      setSearchResult(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSearch = useCallback((query: string) => {
    performSearch(query, 1);
  }, [performSearch]);

  const performBrowse = useCallback(async (nodeId: string, page: number = 1, preserveMatchedNodes: boolean = true) => {
    setLoading(true);
    setError(null);
    setCurrentPage(page);

    // Clear original matched nodes immediately if not preserving
    if (!preserveMatchedNodes) {
      setOriginalMatchedNodes(null);
    }

    try {
      const browseResult = await api.browseNode(nodeId, page, ITEMS_PER_PAGE);
      // Convert BrowseResponse to SearchResponse format
      // Use original matched nodes only if preserving them
      const matchedNodesToUse = preserveMatchedNodes ? originalMatchedNodes : null;
      const result: SearchResponse = {
        query: browseResult.node.name,
        total: browseResult.total,
        page: browseResult.page,
        limit: browseResult.limit,
        taxonomy_path: browseResult.path_to_root,
        matched_nodes: matchedNodesToUse || [{
          id: browseResult.node.id,
          name: browseResult.node.name,
          score: 1.0  // Direct match
        }],
        images: browseResult.images
      };
      setSearchResult(result);
      setCurrentQuery(browseResult.node.name);  // Update query to show browsed node name
      setBrowsingNodeId(nodeId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Browse failed');
      setSearchResult(null);
    } finally {
      setLoading(false);
    }
  }, [originalMatchedNodes]);

  const handlePageChange = useCallback((page: number) => {
    if (browsingNodeId) {
      // In browse mode - use browse API
      performBrowse(browsingNodeId, page);
    } else {
      // In search mode - use search API
      performSearch(currentQuery, page);
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [browsingNodeId, currentQuery, performBrowse, performSearch]);

  const handleMatchedNodeClick = useCallback((nodeId: string) => {
    // Browse a matched category - keep original matched nodes for switching
    performBrowse(nodeId, 1);
  }, [performBrowse]);

  const handleTaxonomyPathClick = useCallback((nodeId: string) => {
    // Browse a taxonomy path node - clear original context and start fresh
    performBrowse(nodeId, 1, false);  // Don't preserve matched nodes
  }, [performBrowse]);

  const handleTagClick = useCallback((tag: string) => {
    performSearch(tag, 1);
  }, [performSearch]);

  const totalPages = searchResult
    ? Math.ceil(searchResult.total / ITEMS_PER_PAGE)
    : 0;

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center gap-4 mb-4">
            <h1 className="text-xl font-bold text-slate-800">
              Semantic Image Browser
            </h1>
            {stats && (
              <div className="text-sm text-slate-500">
                {Math.round(stats.taxonomy.total_nodes / 1000)}k nodes
                {stats.images.total_images > 0 && (
                  <> | {stats.images.total_images.toLocaleString()} images</>
                )}
              </div>
            )}
          </div>

          <SearchBar
            onSearch={handleSearch}
            initialQuery={currentQuery}
          />
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* Error message */}
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-6">
            {error}
          </div>
        )}

        {/* Taxonomy breadcrumb */}
        {searchResult && (
          <TaxonomyBreadcrumb
            path={searchResult.taxonomy_path}
            matchedNodes={searchResult.matched_nodes}
            onPathNodeClick={handleTaxonomyPathClick}
            onMatchedNodeClick={handleMatchedNodeClick}
            selectedNodeId={browsingNodeId}
          />
        )}

        {/* Stats bar */}
        {(searchResult || currentQuery) && (
          <StatsBar
            query={currentQuery}
            total={searchResult?.total || 0}
            page={currentPage}
            limit={ITEMS_PER_PAGE}
            primaryMatchedNode={
              browsingNodeId
                ? searchResult?.matched_nodes?.find(n => n.id === browsingNodeId) || searchResult?.matched_nodes?.[0]
                : searchResult?.matched_nodes?.[0]
            }
          />
        )}

        {/* Image grid */}
        <MasonryGrid
          images={searchResult?.images || []}
          onTagClick={handleTagClick}
          loading={loading}
        />

        {/* Pagination */}
        {searchResult && searchResult.total > 0 && (
          <Pagination
            page={currentPage}
            totalPages={totalPages}
            onPageChange={handlePageChange}
          />
        )}

        {/* Empty state */}
        {!loading && !searchResult && !error && (
          <div className="text-center py-20">
            <svg
              className="w-20 h-20 mx-auto text-slate-300 mb-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            <h2 className="text-2xl font-semibold text-slate-600 mb-2">
              Semantic Image Search
            </h2>
            <p className="text-slate-500 max-w-md mx-auto mb-6">
              Search through images using natural language. Queries are matched
              to the WikiKG taxonomy hierarchy for semantic understanding.
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {['golden retriever', 'sports car', 'oak tree', 'laptop computer'].map(example => (
                <button
                  key={example}
                  onClick={() => handleSearch(example)}
                  className="px-4 py-2 bg-white border border-slate-200 rounded-full text-sm
                             text-slate-600 hover:bg-slate-50 hover:border-slate-300
                             transition-colors duration-150"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 py-6 text-center text-sm text-slate-500">
          Semantic Image Browser
        </div>
      </footer>
    </div>
  );
}

export default App;
