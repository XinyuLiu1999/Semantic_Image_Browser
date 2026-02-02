import type { MatchedNode } from '../types';

interface StatsBarProps {
  query: string;
  total: number;
  page: number;
  limit: number;
  primaryMatchedNode?: MatchedNode;
}

export function StatsBar({ query, total, page, limit, primaryMatchedNode }: StatsBarProps) {
  const start = (page - 1) * limit + 1;
  const end = Math.min(page * limit, total);

  // Use the primary matched node name if available, otherwise fall back to query
  const displayTerm = primaryMatchedNode?.name || query;

  return (
    <div className="flex items-center justify-between mb-4 text-sm text-slate-600">
      <div>
        {total > 0 ? (
          <>
            Showing <span className="font-medium">{start}-{end}</span> of{' '}
            <span className="font-medium">{total}</span> results
            {displayTerm && (
              <>
                {' '}for "<span className="font-medium text-blue-600">{displayTerm}</span>"
              </>
            )}
          </>
        ) : (
          query ? (
            <>No results for "<span className="font-medium">{query}</span>"</>
          ) : (
            'Enter a search query to find images'
          )
        )}
      </div>
    </div>
  );
}
