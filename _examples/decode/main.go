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

	fmt.Println(string(encoding.Decode([]uint{9906, 4435})))
}
