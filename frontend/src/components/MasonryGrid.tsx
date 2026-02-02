import { ImageCard } from './ImageCard';
import type { ImageData } from '../types';

interface MasonryGridProps {
  images: ImageData[];
  onTagClick?: (tag: string) => void;
  loading?: boolean;
}

export function MasonryGrid({ images, onTagClick, loading }: MasonryGridProps) {
  if (loading) {
    return (
      <div className="masonry-grid">
        {Array.from({ length: 12 }).map((_, i) => (
          <div key={i} className="masonry-item">
            <div className="bg-white rounded-xl overflow-hidden shadow-md border border-slate-100">
              <div className="aspect-square loading-shimmer" />
              <div className="p-3 space-y-2">
                <div className="h-4 loading-shimmer rounded" />
                <div className="h-4 loading-shimmer rounded w-2/3" />
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="text-center py-16">
        <svg
          className="w-16 h-16 mx-auto text-slate-300 mb-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1.5}
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
        <p className="text-slate-500 text-lg">No images found</p>
        <p className="text-slate-400 text-sm mt-1">Try a different search query</p>
      </div>
    );
  }

  return (
    <div className="masonry-grid">
      {images.map((image) => (
        <ImageCard
          key={image.id}
          image={image}
          onTagClick={onTagClick}
        />
      ))}
    </div>
  );
}
