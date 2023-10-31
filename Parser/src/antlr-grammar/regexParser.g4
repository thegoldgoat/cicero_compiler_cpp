parser grammar regexParser;
options {
	tokenVocab = regexLexer;
}

// Parser root context, ensures all input is matched
root:
	noprefix=HAT? LPAR regExp RPAR nosuffix=DOLLAR? EOF
	| regExp EOF;

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
	| LBRACKET GROUP_HAT group+ RBRACKET
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

group_terminal_sequence:
	GROUP_CHAR
	| GROUP_DIGIT_CHAR;

group_metachar:
	GROUP_WHITESPACE
	| GROUP_WHITESPACE_COMPLEMENTED
	| GROUP_WORD
	| GROUP_WORD_COMPLEMENTED
	| GROUP_DIGIT
	| GROUP_DIGIT_COMPLEMENTED;

group:
	first_char=group_terminal_sequence MINUS second_char=group_terminal_sequence
	| single_char=group_terminal_sequence
	| group_metachar;