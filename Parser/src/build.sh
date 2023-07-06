antlr4 -o antlrlib -Dlanguage=Cpp regexLexer.g4
antlr4 -o antlrlib -lib antlrlib -Dlanguage=Cpp -no-listener -visitor regexParser.g4