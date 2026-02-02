// Type definitions for the ImageNet21K Viewer

export interface TaxonomyNode {
  id: string;
  name: string;
  synonyms?: string[];
  definition?: string;
  node_type?: string;  // 'category' or 'page'
  depth?: number;
}

export interface ImageData {
  id: string;
  image_path: string;
  generated_caption: string;
  tags: string[];
  relevance_score?: number;
  matched_node?: string;
  linked_nodes?: [string, number][];
  [key: string]: unknown; // Extra metadata
}

export interface MatchedNode {
  id: string;
  name: string;
  score: number;
  parent_name?: string;
}

export interface SearchResponse {
  query: string;
  total: number;
  page: number;
  limit: number;
  taxonomy_path: TaxonomyNode[];
  matched_nodes: MatchedNode[];
  images: ImageData[];
}

export interface TaxonomyNodeDetails {
  node: TaxonomyNode;
  path_to_root: TaxonomyNode[];
  children: TaxonomyNode[];
  image_count: number;
}

export interface BrowseResponse {
  node: TaxonomyNode;
  path_to_root: TaxonomyNode[];
  total: number;
  page: number;
  limit: number;
  images: ImageData[];
}

export interface Suggestion {
  id: string;
  name: string;
}

export interface SystemStats {
  taxonomy: {
    total_nodes: number;
    leaf_nodes: number;
    internal_nodes: number;
    category_nodes: number;
    page_nodes: number;
    max_depth: number;
    root: string;
  };
  images: {
    total_images: number;
    unique_tags: number;
    linked_nodes: number;
    total_links: number;
  };
}
