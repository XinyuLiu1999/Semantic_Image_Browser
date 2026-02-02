import type { TaxonomyNode, MatchedNode } from '../types';

interface TaxonomyBreadcrumbProps {
  path: TaxonomyNode[];
  matchedNodes?: MatchedNode[];
  onPathNodeClick?: (nodeId: string) => void;  // For taxonomy path clicks
  onMatchedNodeClick?: (nodeId: string) => void;  // For matched category clicks
  selectedNodeId?: string | null;  // Currently selected/browsing node
}

export function TaxonomyBreadcrumb({ path, matchedNodes, onPathNodeClick, onMatchedNodeClick, selectedNodeId }: TaxonomyBreadcrumbProps) {
  if (path.length === 0 && (!matchedNodes || matchedNodes.length === 0)) {
    return null;
  }

  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 mb-6 shadow-sm">
      {/* Taxonomy Path */}
      {path.length > 0 && (
        <div className="mb-3">
          <div className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">
            Taxonomy Path
          </div>
          <nav className="flex flex-wrap items-center gap-1">
            {path.map((node, index) => (
              <span key={node.id} className="flex items-center">
                {index > 0 && (
                  <svg
                    className="w-4 h-4 text-slate-300 mx-1"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                )}
                <button
                  onClick={() => onPathNodeClick?.(node.id)}
                  className={`breadcrumb-item px-2 py-1 rounded text-sm
                             ${index === path.length - 1
                               ? 'bg-blue-100 text-blue-700 font-medium'
                               : 'text-slate-600 hover:bg-slate-100'
                             }`}
                  title={node.definition || node.name}
                >
                  {node.name}
                </button>
              </span>
            ))}
          </nav>
        </div>
      )}

      {/* Matched Nodes */}
      {matchedNodes && matchedNodes.length > 0 && (
        <div>
          <div className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">
            Matched Nodes
          </div>
          <div className="flex flex-wrap gap-2">
            {matchedNodes.map((node, index) => {
              // Highlight: selected node, or first node if nothing selected
              const isHighlighted = selectedNodeId ? node.id === selectedNodeId : index === 0;
              return (
                <button
                  key={node.id}
                  onClick={() => onMatchedNodeClick?.(node.id)}
                  className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm
                             transition-colors duration-150
                             ${isHighlighted
                               ? 'bg-green-100 text-green-700 hover:bg-green-200 ring-2 ring-green-400'
                               : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                             }`}
                  title={node.parent_name ? `Parent: ${node.parent_name}` : undefined}
                >
                  <span className="font-medium">
                    {node.name}
                    {node.parent_name && (
                      <span className={`font-normal ${isHighlighted ? 'text-green-600' : 'text-slate-400'}`}>
                        {' '}({node.parent_name})
                      </span>
                    )}
                  </span>
                  <span className={`text-xs ${isHighlighted ? 'text-green-500' : 'text-slate-400'}`}>
                    {(node.score * 100).toFixed(0)}%
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
