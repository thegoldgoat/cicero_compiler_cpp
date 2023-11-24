lexer grammar regexLexer;

@lexer::members {
	std::string decodeEscapedHex(std::string s) {
		std::stringstream ss;
		ss << std::hex << s.substr(2);
		int result;
		ss >> result;
		return std::string(1, static_cast<char>(result));
	}
}

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

ESCAPED_HEX:
	'\\x' [0-9][0-9] { setText(decodeEscapedHex(getText())); };
ESCAPED_CHAR: '\\' . { setText(getText().substr(1)); };

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