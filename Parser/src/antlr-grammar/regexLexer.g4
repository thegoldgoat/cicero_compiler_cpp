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

// Escapes taken from https://docs.python.org/3/reference/lexical_analysis.html#escape-sequences

// Escape a char expressed in as hex digit. "\xb3" for example is "0xb3"
ESCAPED_HEX:
	'\\x' [0-9][0-9] { setText(decodeEscapedHex(getText())); };

// Other escapes
ESCAPE_BELL: '\\a' { setText("\a"); };
ESCAPE_BACKSPACE: '\\b' { setText("\b"); };
ESCAPE_FORMFEED: '\\f' { setText("\f"); };
ESCAPE_NEWLINE: '\\n' { setText("\n"); };
ESCAPE_CARRIAGE_RETURN: '\\r' { setText("\r"); };
ESCAPE_TAB: '\\t' { setText("\t"); };
ESCAPE_VERTICAL_TAB: '\\v' { setText("\v"); };

// Most general escape
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

GROUP_ESCAPED_HEX:
	'\\x' [0-9][0-9] { setText(decodeEscapedHex(getText())); };

GROUP_ESCAPE_BELL: '\\a' { setText("\a"); };
GROUP_ESCAPE_BACKSPACE: '\\b' { setText("\b"); };
GROUP_ESCAPE_FORMFEED: '\\f' { setText("\f"); };
GROUP_ESCAPE_NEWLINE: '\\n' { setText("\n"); };
GROUP_ESCAPE_CARRIAGE_RETURN: '\\r' { setText("\r"); };
GROUP_ESCAPE_TAB: '\\t' { setText("\t"); };
GROUP_ESCAPE_VERTICAL_TAB: '\\v' { setText("\v"); };

GROUP_ESCAPED_CHAR: '\\' . { setText(getText().substr(1)); };

RBRACKET: ']' -> popMode;


mode QUANTITY;
QUANTITY_END: '}' -> popMode;
QUANTITY_NUM: [0-9]+;
QUANTITY_SEPARATOR: ',';