import { useState } from 'react';
import { api } from '../api';
import type { ImageData } from '../types';

interface ImageCardProps {
  image: ImageData;
  onTagClick?: (tag: string) => void;
}

export function ImageCard({ image, onTagClick }: ImageCardProps) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [showMetadata, setShowMetadata] = useState(false);

  const imageUrl = api.getImageUrl(image.id);

  // Extract extra metadata (exclude standard fields)
  const extraFields = Object.entries(image).filter(
    ([key]) => !['id', 'image_path', 'generated_caption', 'tags', 'relevance_score', 'matched_node', 'linked_nodes'].includes(key)
  );

  return (
    <div
      className="masonry-item"
      onMouseEnter={() => setShowMetadata(true)}
      onMouseLeave={() => setShowMetadata(false)}
    >
      <div className="image-card relative bg-white rounded-xl overflow-hidden shadow-md border border-slate-100">
        {/* Image */}
        <div className="relative aspect-square">
          {!isLoaded && !hasError && (
            <div className="absolute inset-0 loading-shimmer" />
          )}

          {hasError ? (
            <div className="absolute inset-0 flex items-center justify-center bg-slate-100 text-slate-400">
              <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                      d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
          ) : (
            <img
              src={imageUrl}
              alt={image.generated_caption || image.id}
              className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-300
                         ${isLoaded ? 'opacity-100' : 'opacity-0'}`}
              onLoad={() => setIsLoaded(true)}
              onError={() => setHasError(true)}
              loading="lazy"
            />
          )}

          {/* Relevance score badge */}
          {image.relevance_score !== undefined && (
            <div className="absolute top-2 right-2 px-2 py-1 bg-black/60 text-white text-xs rounded-full font-medium">
              {(image.relevance_score * 100).toFixed(0)}%
            </div>
          )}
        </div>

        {/* Caption (always visible) */}
        {image.generated_caption && (
          <div className="p-3 border-t border-slate-100">
            <p className="text-sm text-slate-600 line-clamp-2">
              {image.generated_caption}
            </p>
          </div>
        )}

        {/* Hover metadata overlay */}
        {showMetadata && (
          <div className="absolute inset-0 bg-black/85 text-white p-4 overflow-y-auto
                          animate-in fade-in duration-200">
            <div className="space-y-3">
              {/* ID */}
              <div>
                <div className="text-xs font-medium text-slate-400 uppercase">ID</div>
                <div className="text-sm font-mono">{image.id}</div>
              </div>

              {/* Path */}
              <div>
                <div className="text-xs font-medium text-slate-400 uppercase">Path</div>
                <div className="text-xs font-mono text-slate-300 break-all">
                  {image.image_path}
                </div>
              </div>

              {/* Tags */}
              {image.tags && image.tags.length > 0 && (
                <div>
                  <div className="text-xs font-medium text-slate-400 uppercase mb-1">Tags</div>
                  <div className="flex flex-wrap gap-1">
                    {image.tags.map((tag, index) => (
                      <button
                        key={index}
                        onClick={(e) => {
                          e.stopPropagation();
                          onTagClick?.(tag);
                        }}
                        className="px-2 py-0.5 bg-blue-500/30 text-blue-200 rounded text-xs
                                   hover:bg-blue-500/50 transition-colors"
                      >
                        {tag}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Caption */}
              {image.generated_caption && (
                <div>
                  <div className="text-xs font-medium text-slate-400 uppercase">Caption</div>
                  <div className="text-sm">{image.generated_caption}</div>
                </div>
              )}

              {/* Extra metadata */}
              {extraFields.length > 0 && (
                <div>
                  <div className="text-xs font-medium text-slate-400 uppercase mb-1">Metadata</div>
                  <div className="space-y-1">
                    {extraFields.map(([key, value]) => (
                      <div key={key} className="text-xs">
                        <span className="text-slate-400">{key}:</span>{' '}
                        <span className="text-slate-200">
                          {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Matched node */}
              {image.matched_node && (
                <div>
                  <div className="text-xs font-medium text-slate-400 uppercase">Matched Node</div>
                  <div className="text-xs font-mono text-green-300">{image.matched_node}</div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
