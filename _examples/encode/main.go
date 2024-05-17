package main

import (
	"fmt"
	"log"

	"github.com/hupe1980/go-tiktoken"
)

func main() {
	encoding, err := tiktoken.NewEncodingForModel("gpt-3.5-turbo")
	if err != nil {
		log.Fatal(err)
	}

	ids, tokens, err := encoding.Encode("Hello World", nil, nil)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("IDs:", ids)
	fmt.Println("Tokens:", tokens)
}
