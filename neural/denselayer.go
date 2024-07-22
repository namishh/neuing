package neural

import (
	"math/rand"
)

// make a Dense layer struct
type DenseLayer struct {
	dWeights       [][]float64
	dBias          [][]float64
	dInputs        [][]float64
	weights        [][]float64
	bias           [][]float64
	output         [][]float64
	inputs         [][]float64
	biasMomentum   [][]float64
	weightMomentum [][]float64
	weightCache    [][]float64
	biasCache      [][]float64
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
	bias := zeros(1, 64)
	return &DenseLayer{bias: bias, weights: Initweights(nInputs, nNeurons, 0.01)}
}

func (d *DenseLayer) Forward(inputs [][]float64) {
	mp := matrixProduct(inputs, d.weights)
	for i := 0; i < len(mp); i++ {
		for j := 0; j < len(mp[i]); j++ {
			mp[i][j] += d.bias[0][j]
		}
	}
	d.output = mp
	d.inputs = inputs
}

func Sum(matrix [][]float64) [][]float64 {
	numRows := len(matrix)
	numCols := len(matrix[0])
	result := make([][]float64, 1)
	result[0] = make([]float64, numCols)
	for j := 0; j < numCols; j++ {
		sum := 0.0
		for i := 0; i < numRows; i++ {
			sum += matrix[i][j]
		}
		result[0][j] = sum
	}
	return result
}

func (d *DenseLayer) Backward(dValues [][]float64) {

	d.dWeights = matrixProduct(transpose(d.inputs), dValues)
	d.dBias = Sum(transpose(dValues))
	d.dInputs = matrixProduct(dValues, transpose(d.weights))
}
