package main

import (
	"fmt"
	"math"
	"math/rand"
)

// this is just generated from chatgpt which converts
// https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/spiral.py this code to this
// orignal license : https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE
func generateData(samples, classes int) (X [][]float64, y []int) {
	X = make([][]float64, samples*classes)
	y = make([]int, samples*classes)
	for classNumber := 0; classNumber < classes; classNumber++ {
		ixStart := samples * classNumber
		r := make([]float64, samples)
		t := make([]float64, samples)
		for i := 0; i < samples; i++ {
			r[i] = float64(i) / float64(samples-1)
			t[i] = float64(classNumber)*4.0 + float64(classNumber+1)*4.0*float64(i)/float64(samples) + rand.NormFloat64()*0.2
			X[ixStart+i] = []float64{r[i] * math.Sin(t[i]*2.5), r[i] * math.Cos(t[i]*2.5)}
			y[ixStart+i] = classNumber
		}
	}
	return
}

// transposition of a matrix ; rotating a matrix by 90 degrees
// https://en.wikipedia.org/wiki/Transpose
func transpose(m [][]float64) [][]float64 {
	if len(m) == 0 {
		return m
	}
	r := len(m)
	c := len(m[0])
	t := make([][]float64, c)
	for i := 0; i < c; i++ {
		t[i] = make([]float64, r)
		for j := 0; j < r; j++ {
			t[i][j] = m[j][i]
		}
	}
	return t
}

// standard product of two matrices
func matrixProduct(m1, m2 [][]float64) [][]float64 {
	r1 := len(m1)
	c1 := len(m1[0])
	r2 := len(m2)
	c2 := len(m2[0])

	if c1 != r2 {
		panic("Matrices can't be multiplied")
	}

	result := make([][]float64, r1)
	for i := 0; i < r1; i++ {
		result[i] = make([]float64, c2)
		for j := 0; j < c2; j++ {
			for k := 0; k < c1; k++ {
				result[i][j] += m1[i][k] * m2[k][j]
			}
		}
	}
	return result
}

func addBias(p [][]float64, bias []float64) (output [][]float64) {
	l := len(p)
	if l == 0 {
		return nil
	}
	if l != len(bias) {
		panic("Can't Add Bias: Number of biases must match the number of slices in p")
	}
	output = make([][]float64, l)
	for i := 0; i < l; i++ {
		output[i] = make([]float64, len(p[i]))
		for j := 0; j < len(p[i]); j++ {
			output[i][j] = p[i][j] + bias[i]
		}
	}
	return
}

func getOutput(inputs [][]float64, weights [][]float64, bias []float64) [][]float64 {
	return addBias(matrixProduct(inputs, transpose(weights)), bias)
}

// make a Dense layer struct
type DenseLayer struct {
	weights [][]float64
	bias    [][]float64
}

// implmentation of numpy.zeros
func zeros(x int, y int) [][]float64 {
	grid := make([][]float64, x)
	for i := 0; i < x; i++ {
		grid[i] = make([]float64, y)
		for j := 0; j < y; j++ {
			grid[i][j] = 0
		}
	}
	return grid
}

func InitWeights(nInputs int, nNeurons int, multiplier float64) [][]float64 {
	weights := make([][]float64, nInputs)
	for i := 0; i < nInputs; i++ {
		weights[i] = make([]float64, nNeurons)
		for j := 0; j < nNeurons; j++ {
			weights[i][j] = multiplier * rand.NormFloat64()
		}
	}
	return weights
}

// make a Dense layer constructor
func NewDenseLayer(nInputs int, nNeurons int) *DenseLayer {
	bias := zeros(1, 5)
	return &DenseLayer{bias: bias, weights: InitWeights(nInputs, nNeurons, 0.01)}
}

func main() {
	fmt.Println("Hello World!")
	// X, y := generateData(100, 3)
	// fmt.Println(X, y)

	x := NewDenseLayer(2, 3)
	fmt.Println(x.weights)
}
