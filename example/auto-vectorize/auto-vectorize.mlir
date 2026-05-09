#matmul_accesses = [
  affine_map<(m, n, k) -> (m, k)>,
  affine_map<(m, n, k) -> (k, n)>,
  affine_map<(m, n, k) -> (m, n)>
]
#matmul_trait = {
  indexing_maps = #matmul_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

module {
  func.func @auto_vectorize_example(
      %q: vector<2x4xf32>,
      %k: vector<4x8xf32>,
      %v: vector<8x4xf32>,
      %score_init: vector<2x8xf32>,
      %out_init: vector<2x4xf32>) -> vector<2x4xf32> {
    %scores = vector.contract #matmul_trait %q, %k, %score_init
        : vector<2x4xf32>, vector<4x8xf32> into vector<2x8xf32>
    %probs = math.exp %scores : vector<2x8xf32>
    %out = vector.contract #matmul_trait %probs, %v, %out_init
        : vector<2x8xf32>, vector<8x4xf32> into vector<2x4xf32>
    return %out : vector<2x4xf32>
  }
}
