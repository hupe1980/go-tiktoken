package tiktoken

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestClaude(t *testing.T) {
	claude, err := NewClaude()
	require.NoError(t, err)

	encoding, err := NewEncoding(claude)
	require.NoError(t, err)

	t.Run("small text", func(t *testing.T) {
		idx, _ := encoding.EncodeOrdinary("hello world!")
		require.Equal(t, 3, len(idx))
	})

	t.Run("small text", func(t *testing.T) {
		idx, _ := encoding.EncodeOrdinary("hello world!")
		require.Equal(t, 3, len(idx))
	})

	t.Run("text normalising", func(t *testing.T) {
		idx, _ := encoding.EncodeOrdinary("™")
		assert.Equal(t, 1, len(idx))

		idx, _ = encoding.EncodeOrdinary("ϰ")
		assert.Equal(t, 1, len(idx))
	})

	t.Run("allows special tokens", func(t *testing.T) {
		idx, _, err := encoding.Encode("<EOT>", AllSpecial, nil)
		require.NoError(t, err)
		require.Equal(t, 1, len(idx))
	})
}
