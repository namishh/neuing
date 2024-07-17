package neural

type SGDOptimizer struct {
	learningRate float64
}

func NewSGDOptimizer(learningRate float64) *SGDOptimizer {
	return &SGDOptimizer{learningRate: learningRate}
}

func (s *SGDOptimizer) Update(layer *DenseLayer) {
	for i := range layer.weights {
		for j := range layer.weights[i] {
			layer.weights[i][j] += -s.learningRate * layer.dWeights[i][j]
		}
	}
	for i := range layer.bias {
		for j := range layer.bias[i] {
			layer.bias[i][j] += -s.learningRate * layer.dBias[i][j]
		}
	}
}
