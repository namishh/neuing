package neural

import "math/rand"

// make a Dense layer struct
type DenseLayer struct {
	dWeights [][]float64
	dBias    [][]float64
	dInputs  [][]float64
	weights  [][]float64
	bias     [][]float64
	output   [][]float64
}

func Initweights(nInputs int, nNeurons int, multiplier float64) [][]float64 {
	weights := make([][]float64, nInputs)
	for i := 0; i < nInputs; i++ {
		weights[i] = make([]float64, nNeurons)
		for j := 0; j < nNeurons; j++ {
			weights[i][j] = multiplier * rand.NormFloat64()
		}
	}
	return weights
}

func NewDenseLayer(nInputs int, nNeurons int) *DenseLayer {
	bias := zeros(1, 5)
	return &DenseLayer{bias: bias, weights: Initweights(nInputs, nNeurons, 0.01)}
}

func (d *DenseLayer) Forward(inputs [][]float64) {
	mp := matrixProduct(inputs, d.weights)
	for ind, arr := range mp {
		for idx := range arr {
			mp[ind][idx] += d.bias[0][idx]
		}
	}
	d.output = mp
	d.dInputs = inputs
}

func (d *DenseLayer) Backward(dValues [][]float64) {
	d.dWeights = matrixProduct(transpose(d.dInputs), dValues)
	a := transpose(dValues)
	d.dBias = make([][]float64, 1)
	d.dBias[0] = make([]float64, len(a[0]))
	for i := 0; i < len(a); i++ {
		for j := 0; j < len(a[i]); j++ {
			d.dBias[0][j] += a[i][j]
		}
	}
	d.dInputs = matrixProduct(dValues, transpose(d.weights))
}
