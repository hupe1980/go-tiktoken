package tiktoken

import (
	"testing"

	"github.com/dlclark/regexp2"
	"github.com/stretchr/testify/assert"
)

func TestGPT2Encoding(t *testing.T) {
	encoding, err := NewEncodingByName(GPT2)
	assert.NoError(t, err)

	t.Run("default", func(t *testing.T) {
		text := "hello world"
		ids, _ := encoding.EncodeOrdinary(text)
		assert.ElementsMatch(t, []uint{31373, 995}, ids)
	})

	t.Run("special token", func(t *testing.T) {
		text := "hello <|endoftext|>"
		ids, _, err := encoding.Encode(text, []string{"all"}, nil)
		assert.NoError(t, err)
		assert.ElementsMatch(t, []uint{31373, 220, 50256}, ids)
	})

	t.Run("decode", func(t *testing.T) {
		assert.Equal(t, "hello world", string(encoding.Decode([]uint{31373, 995})))
	})

	t.Run("not allowed", func(t *testing.T) {
		text := "hello <|endoftext|>"
		_, _, err := encoding.Encode(text, nil, []string{"<|endoftext|>"})
		assert.Error(t, err)
	})
}

func TestCL100kEncoding(t *testing.T) {
	encoding, err := NewEncodingByName(CL100kBase)
	assert.NoError(t, err)

	t.Run("default", func(t *testing.T) {
		text := "hello world"
		ids, _ := encoding.EncodeOrdinary(text)
		assert.ElementsMatch(t, []uint{15339, 1917}, ids)
	})

	t.Run("special token", func(t *testing.T) {
		text := "hello <|endoftext|>"
		ids, _, err := encoding.Encode(text, []string{"all"}, nil)
		assert.NoError(t, err)
		assert.ElementsMatch(t, []uint{15339, 220, 100257}, ids)
	})

	t.Run("decode", func(t *testing.T) {
		assert.Equal(t, "hello world", string(encoding.Decode([]uint{15339, 1917})))
	})

	t.Run("not allowed", func(t *testing.T) {
		text := "hello <|endoftext|>"
		_, _, err := encoding.Encode(text, nil, []string{"<|endoftext|>"})
		assert.Error(t, err)
	})
}

func TestNewEncodingForModel(t *testing.T) {
	enc, err := NewEncodingForModel("gpt2")
	assert.NoError(t, err)
	assert.Equal(t, GPT2, enc.Name())

	enc, err = NewEncodingForModel("text-davinci-003")
	assert.NoError(t, err)
	assert.Equal(t, P50kBase, enc.Name())

	enc, err = NewEncodingForModel("text-davinci-edit-001")
	assert.NoError(t, err)
	assert.Equal(t, P50kEdit, enc.Name())

	enc, err = NewEncodingForModel("gpt-3.5-turbo-0301")
	assert.NoError(t, err)
	assert.Equal(t, CL100kBase, enc.Name())
}

func TestSpecialTokenRegex(t *testing.T) {
	testCases := []struct {
		name                 string
		disallowedSpecialSet map[string]any
		expectedRegex        string
	}{
		{
			name: "Single special token",
			disallowedSpecialSet: map[string]any{
				"special1": struct{}{},
			},
			expectedRegex: "special1",
		},
		{
			name: "Multiple special tokens",
			disallowedSpecialSet: map[string]any{
				"special1": struct{}{},
				"special2": struct{}{},
			},
			expectedRegex: "special1|special2",
		},
		{
			name:                 "Empty set",
			disallowedSpecialSet: map[string]any{},
			expectedRegex:        "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			regex := specialTokenRegex(tc.disallowedSpecialSet)
			assert.Equal(t, tc.expectedRegex, regex.String(), "Unexpected regex pattern")
		})
	}
}

func TestFindRegex2StringMatch(t *testing.T) {
	testCases := []struct {
		name     string
		text     string
		pattern  string
		expected string
	}{
		{
			name:     "Matching pattern",
			text:     "Hello, world!",
			pattern:  "world",
			expected: "world",
		},
		{
			name:     "Non-matching pattern",
			text:     "Hello, world!",
			pattern:  "foo",
			expected: "",
		},
		{
			name:     "Empty text",
			text:     "",
			pattern:  "bar",
			expected: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			reg := regexp2.MustCompile(tc.pattern, 0)
			result := findRegex2StringMatch(tc.text, reg)
			assert.Equal(t, tc.expected, result, "Unexpected result")
		})
	}
}

func TestDifference(t *testing.T) {
	testCases := []struct {
		name   string
		setA   map[string]any
		setB   map[string]any
		result map[string]any
	}{
		{
			name:   "Set A is empty",
			setA:   map[string]any{},
			setB:   map[string]any{"key1": true, "key2": true},
			result: map[string]any{},
		},
		{
			name:   "Set B is empty",
			setA:   map[string]any{"key1": true, "key2": true},
			setB:   map[string]any{},
			result: map[string]any{"key1": true, "key2": true},
		},
		{
			name:   "No common keys",
			setA:   map[string]any{"key1": true, "key2": true},
			setB:   map[string]any{"key3": true, "key4": true},
			result: map[string]any{"key1": true, "key2": true},
		},
		{
			name:   "Some common keys",
			setA:   map[string]any{"key1": true, "key2": true, "key3": true},
			setB:   map[string]any{"key2": true, "key4": true},
			result: map[string]any{"key1": true, "key3": true},
		},
		{
			name:   "All keys are common",
			setA:   map[string]any{"key1": true, "key2": true},
			setB:   map[string]any{"key1": true, "key2": true},
			result: map[string]any{},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := difference(tc.setA, tc.setB)
			assert.Equal(t, tc.result, result, "Unexpected result")
		})
	}
}
