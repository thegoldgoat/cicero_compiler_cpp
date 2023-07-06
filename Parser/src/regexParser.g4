parser grammar regexParser;
options {
	tokenVocab = regexLexer;
}

// Parser root context, ensures all input is matched
root: regExp EOF;

// Regular Expression, alternatives of concatenations
regExp: concatenation (PIPE concatenation)*;

// Concatenation, list of pieces
concatenation: pieces+=piece+;

piece: atom quantifier?;

quantifier:
	QUESTION
	| STAR
	| PLUS
	| LBRACE quantity RBRACE;
quantity:
	minnum=NUM COMMA maxnum=NUM // From Integer_1 to Integer_2
	| atleastnum=NUM COMMA // At least Integer
	| exactlynum=NUM; // Exactly Integer

atom:
	LPAR regExp RPAR
	| LBRACKET group RBRACKET
	| LBRACKET HAT group RBRACKET
	| terminal_sequence
	| metachar
	| ANYCHAR
	| DOLLAR
	| MINUS;

terminal_sequence: CHAR | NUM | COMMA;

metachar:
	WHITESPACE
	| WHITESPACE_COMPLEMENTED
	| WORD
	| WORD_COMPLEMENTED
	| DIGIT
	| DIGIT_COMPLEMENTED;

group:
	terminal_sequence MINUS terminal_sequence group
	| terminal_sequence MINUS terminal_sequence
	| terminal_sequence group
	| terminal_sequence
	| metachar group
	| metachar;