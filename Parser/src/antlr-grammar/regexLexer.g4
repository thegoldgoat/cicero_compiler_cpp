lexer grammar regexLexer;

PIPE: '|';
STAR: '*';
PLUS: '+';
MINUS: '-';
QUESTION: '?';
DOLLAR: '$';
ANYCHAR: '.';
HAT: '^';
COMMA: ',';

LPAR: '(';
RPAR: ')';
LBRACKET: '[';
RBRACKET: ']';
LBRACE: '{';
RBRACE: '}';

NUM: [0-9]+;

DIGIT: '\\d';
DIGIT_COMPLEMENTED: '\\D';

WORD: '\\w';
WORD_COMPLEMENTED: '\\W';

WHITESPACE: '\\s';
WHITESPACE_COMPLEMENTED: '\\S';

CHAR:
	~('.' | '\\' | '?' | '*' | '+' | '(' | ')' | '|' | '[' | ']');