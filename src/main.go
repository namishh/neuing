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
	output  [][]float64
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

// make a Dense layer constructor
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
}

type ReLU struct {
	output [][]float64
}

func NewReLU() *ReLU {
	return &ReLU{}
}

// if x > 0 return x else return 0
func (r *ReLU) Forward(inputs [][]float64) {
	r.output = make([][]float64, len(inputs))
	for i, arr := range inputs {
		r.output[i] = make([]float64, len(arr))
		for j, val := range arr {
			r.output[i][j] = math.Max(0, val)
		}
	}
}

type Softmax struct {
	output [][]float64
}

func NewSoftmax() *Softmax {
	return &Softmax{}
}

// replace of numpy.exp
func Exponential(inputs [][]float64) [][]float64 {
	E := math.E
	for i, arr := range inputs {
		for j, val := range arr {
			inputs[i][j] = math.Pow(E, val)
		}
	}
	return inputs
}

func Max(inputs [][]float64) [][]float64 {
	result := make([][]float64, len(inputs))
	for i, row := range inputs {
		maxValue := math.Inf(-1) // negative infinity
		for _, value := range row {
			if value > maxValue {
				maxValue = value
			}
		}
		result[i] = []float64{maxValue}
	}
	return result
}

func (s *Softmax) Forward(inputs [][]float64) {
	// subtract Inputs and Max(Inputs)
	difference := make([][]float64, len(inputs))
	maxInputs := Max(inputs)
	for i, arr := range inputs {
		difference[i] = make([]float64, len(arr))
		for j, val := range arr {
			difference[i][j] = val - maxInputs[i][0]
		}
	}

	exponentials := Exponential(difference)

	rowSums := make([]float64, len(exponentials))
	for i, row := range exponentials {
		sum := 0.0
		for _, value := range row {
			sum += value
		}
		rowSums[i] = sum
	}

	probabilities := make([][]float64, len(exponentials))
	for i, row := range exponentials {
		probRow := make([]float64, len(row))
		for j, value := range row {
			probRow[j] = value / rowSums[i]
		}
		probabilities[i] = probRow
	}

	s.output = probabilities
}

func main() {
	fmt.Println("Hello World!")
	X, _ := generateData(100, 3)
	// fmt.Println(X, y)

	denseLayer1 := NewDenseLayer(2, 3)
	denseLayer1.Forward(X)

	activation := NewReLU()
	activation.Forward(denseLayer1.output)

	denseLayer2 := NewDenseLayer(3, 3)
	denseLayer2.Forward(activation.output)

	softmax := NewSoftmax()
	softmax.Forward(denseLayer2.output)

	fmt.Println(softmax.output)
}
