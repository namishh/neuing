package neural

type SGDOptimizer struct {
	learningRate        float64
	decay               float64
	currentLearningRate float64
	iterations          int
	momentum            float64
}

func NewSGDOptimizer(learningRate float64, decay float64, momentum float64) *SGDOptimizer {
	return &SGDOptimizer{learningRate: learningRate, decay: decay, currentLearningRate: learningRate, iterations: 0, momentum: momentum}
}

func (s *SGDOptimizer) Update(layer *DenseLayer) {
	if len(layer.weightMomentum) == 0 {
		layer.weightMomentum = zeros(len(layer.weights), len(layer.weights[0]))
		layer.biasMomentum = zeros(len(layer.bias), len(layer.bias[0]))
	}

	for i := range layer.weights {
		for j := range layer.weights[i] {
			layer.weights[i][j] += s.momentum*layer.weightMomentum[i][j] - s.currentLearningRate*layer.dWeights[i][j]
			layer.weightMomentum[i][j] = s.momentum*layer.weightMomentum[i][j] - s.currentLearningRate*layer.dWeights[i][j]
		}
	}

	for i := range layer.bias {
		for j := range layer.bias[i] {
			layer.bias[i][j] += s.momentum*layer.biasMomentum[i][j] - s.currentLearningRate*layer.dBias[i][j]
			layer.biasMomentum[i][j] = s.momentum*layer.biasMomentum[i][j] - s.currentLearningRate*layer.dBias[i][j]
		}
	}
}

func (s *SGDOptimizer) PreUpdate() {
	if s.decay != 0 {
		s.currentLearningRate = s.learningRate / (1 + s.decay*float64(s.iterations))
	}
}

func (s *SGDOptimizer) PostUpdate() {
	s.iterations++
}
