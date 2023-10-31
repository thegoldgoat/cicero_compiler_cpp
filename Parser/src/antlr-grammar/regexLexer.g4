lexer grammar regexLexer;

PIPE: '|';
STAR: '*';
PLUS: '+';
QUESTION: '?';
DOLLAR: '$';
ANYCHAR: '.';
HAT: '^';

LPAR: '(';
RPAR: ')';
LBRACKET: '[' -> pushMode(GROUP);
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

ESCAPED_CHAR: '\\' .;

mode GROUP;
GROUP_HAT: '^';
MINUS: '-';
GROUP_CHAR:
	~('.' | '\\' | '?' | '*' | '+' | '(' | ')' | '|' | '[' | ']');
GROUP_DIGIT_CHAR: [0-9];

GROUP_DIGIT: '\\d';
GROUP_DIGIT_COMPLEMENTED: '\\D';
GROUP_WORD: '\\w';
GROUP_WORD_COMPLEMENTED: '\\W';
GROUP_WHITESPACE: '\\s';
GROUP_WHITESPACE_COMPLEMENTED: '\\S';
RBRACKET: ']' -> popMode;


mode QUANTITY;
QUANTITY_END: '}' -> popMode;
QUANTITY_NUM: [0-9]+;
QUANTITY_SEPARATOR: ',';