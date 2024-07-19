package neural

type SGDOptimizer struct {
	learningRate        float64
	decay               float64
	currentLearningRate float64
	iterations          int
}

func NewSGDOptimizer(learningRate float64, decay float64) *SGDOptimizer {
	return &SGDOptimizer{learningRate: learningRate, decay: decay, currentLearningRate: learningRate, iterations: 0}
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

func (s *SGDOptimizer) PreUpdate() {
	s.currentLearningRate = s.learningRate / (1 + s.decay*float64(s.iterations))
}

func (s *SGDOptimizer) PostUpdate() {
	s.iterations++
}
