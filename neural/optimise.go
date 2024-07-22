package neural

import (
	"math"
)

type SGDOptimizer struct {
	learningRate        float64
	decay               float64
	currentLearningRate float64
	iterations          int
	epsilon             float64
	beta1               float64
	beta2               float64
}

func NewSGDOptimizer(learningRate float64, decay float64, epsilon float64) *SGDOptimizer {
	return &SGDOptimizer{learningRate: learningRate, decay: decay, currentLearningRate: learningRate, iterations: 1, epsilon: epsilon, beta1: 0.9, beta2: 0.999}
}

func (s *SGDOptimizer) Update(layer *DenseLayer) {
	if len(layer.weightCache) == 0 {
		layer.weightMomentum = zeros(len(layer.dWeights), len(layer.dWeights[0]))
		layer.weightCache = zeros(len(layer.dWeights), len(layer.dWeights[0]))

		layer.biasMomentum = zeros(len(layer.bias), len(layer.bias[0]))
		layer.biasCache = zeros(len(layer.bias), len(layer.bias[0]))
	}

	// Updating the momentum
	for i := range layer.dWeights {
		for j := range layer.dWeights[i] {
			//		fmt.Println(layer.weightMomentum[i][j], layer.dWeights[i][j])
			layer.weightMomentum[i][j] = s.beta1*layer.weightMomentum[i][j] + (1-s.beta1)*layer.dWeights[i][j]
		}
	}

	for i := range layer.biasMomentum {
		for j := range layer.biasMomentum[i] {
			layer.biasMomentum[i][j] = s.beta1*layer.biasMomentum[i][j] + (1-s.beta1)*layer.dBias[i][j]
		}
	}

	// Correcting the momentum and bias momentum
	correctedWeightMomentum := make([][]float64, len(layer.weightMomentum))
	for i := range layer.weightMomentum {
		correctedWeightMomentum[i] = make([]float64, len(layer.weightMomentum[i]))
		for j := range layer.weightMomentum[i] {
			correctedWeightMomentum[i][j] = layer.weightMomentum[i][j] / (1 - math.Pow(s.beta1, float64(s.iterations)+1.0))
		}
	}

	correctedBiasMomentum := make([][]float64, len(layer.biasMomentum))
	for i := range layer.biasMomentum {
		correctedBiasMomentum[i] = make([]float64, len(layer.biasMomentum[i]))
		for j := range layer.biasMomentum[i] {
			correctedBiasMomentum[i][j] = layer.biasMomentum[i][j] / (1 - math.Pow(s.beta1, float64(s.iterations)+1.0))
		}
	}

	// Updating the cache
	for i := range layer.weightCache {
		for j := range layer.weightCache[i] {
			layer.weightCache[i][j] = s.beta2*layer.weightCache[i][j] + (1-s.beta2)*layer.dWeights[i][j]*layer.dWeights[i][j]
		}
	}

	for i := range layer.biasCache {
		for j := range layer.biasCache[i] {
			layer.biasCache[i][j] = s.beta2*layer.biasCache[i][j] + (1-s.beta2)*layer.dBias[i][j]*layer.dBias[i][j]
		}
	}

	// Correcting the caches

	correctedWeightCache := make([][]float64, len(layer.weightCache))
	for i := range layer.weightCache {
		correctedWeightCache[i] = make([]float64, len(layer.weightCache[i]))
		for j := range layer.weightCache[i] {
			correctedWeightCache[i][j] = layer.weightCache[i][j] / (1 - math.Pow(s.beta2, float64(s.iterations)+1.0))
		}
	}

	correctedBiasCache := make([][]float64, len(layer.biasCache))
	for i := range layer.biasCache {
		correctedBiasCache[i] = make([]float64, len(layer.biasCache[i]))
		for j := range layer.biasCache[i] {
			correctedBiasCache[i][j] = layer.biasCache[i][j] / (1 - math.Pow(s.beta2, float64(s.iterations)+1.0))
		}
	}

	// Updating the weights and bias
	for i := range layer.weights {
		for j := range layer.weights[i] {
			layer.weights[i][j] += -s.currentLearningRate * correctedWeightMomentum[i][j] / (math.Sqrt(correctedWeightCache[i][j]) + s.epsilon)
		}
	}

	for i := range layer.bias {
		for j := range layer.bias[i] {
			layer.bias[i][j] += -s.currentLearningRate * correctedBiasMomentum[i][j] / (math.Sqrt(correctedBiasCache[i][j]) + s.epsilon)
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
