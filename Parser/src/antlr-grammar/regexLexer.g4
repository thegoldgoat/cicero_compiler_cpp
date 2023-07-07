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
QUANTITY_BEGIN: '{' -> pushMode(QUANTITY);

DIGIT_CHAR: [0-9];

DIGIT: '\\d';
DIGIT_COMPLEMENTED: '\\D';

WORD: '\\w';
WORD_COMPLEMENTED: '\\W';

WHITESPACE: '\\s';
WHITESPACE_COMPLEMENTED: '\\S';

CHAR:
	~('.' | '\\' | '?' | '*' | '+' | '(' | ')' | '|' | '[' | ']');

mode QUANTITY;
QUANTITY_END: '}' -> popMode;
QUANTITY_NUM: [0-9]+;
QUANTITY_SEPARATOR: ',';