parser grammar regexParser;
options {
	tokenVocab = regexLexer;
}

// Parser root context, ensures all input is matched
root: noprefix=HAT? regExp nosuffix=DOLLAR? EOF;

// Regular Expression, alternatives of concatenations
regExp: concatenation (PIPE concatenation)*;

// Concatenation, list of pieces
concatenation: pieces+=piece+ DOLLAR?;

piece: atom quantifier?;

quantifier:
	QUESTION
	| STAR
	| PLUS
	| QUANTITY_BEGIN quantity QUANTITY_END;
quantity:
	minnum=QUANTITY_NUM QUANTITY_SEPARATOR maxnum=QUANTITY_NUM // From Integer_1 to Integer_2
	| atleastnum=QUANTITY_NUM QUANTITY_SEPARATOR // At least Integer
	| exactlynum=QUANTITY_NUM; // Exactly Integer

atom:
	LPAR regExp RPAR
	| LBRACKET group+ RBRACKET
	| LBRACKET HAT group+ RBRACKET
	| terminal_sequence
	| metachar
	| ANYCHAR;

terminal_sequence: CHAR | DIGIT_CHAR;

metachar:
	WHITESPACE
	| WHITESPACE_COMPLEMENTED
	| WORD
	| WORD_COMPLEMENTED
	| DIGIT
	| DIGIT_COMPLEMENTED;

group:
	first_char=terminal_sequence MINUS second_char=terminal_sequence
	| single_char=terminal_sequence
	| metachar;