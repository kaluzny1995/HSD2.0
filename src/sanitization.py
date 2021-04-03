import re
import csv

import emot
import emoji


class TweetSanitizer:
    def __init__(self):
        self.url_pat = '(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|http[s]?://)'
        self.mention_pat = '@[\w\-]+'
        self.photo_pat = 'pic.twitter.com\/[\w\-]+'
        self.hashcode_pat = '#.[a-zA-Z0-9]{11}.*(twitter|facebook|reddit|youtube)'
        self.hash_pat = '#'
        self.multidot_pat = '(\.\.\.|…)'
        self.black_sq_pat = '■'
        self.space_pat = '\s+'
    
    def __str__(self):
        return 0

    def partial_sanitization(self, text):
        text = re.sub(self.url_pat, '', text)
        text = re.sub(self.mention_pat, '', text)
        text = re.sub(self.photo_pat, '', text)
        text = re.sub(self.hashcode_pat, '', text)
        text = re.sub(self.multidot_pat, '', text)
        text = re.sub(self.black_sq_pat, '', text)
        text = re.sub(self.space_pat, ' ', text)

        return text

    def extract_emoji___(self, text):
        desc = emot.emoji(text)
        emojis = ' '.join(desc['value'])

        for emoji_ in set(desc['value']):
            text = re.sub(emoji_, '', text)

        return text, emojis

    def extract_emoji(self, text):
        emoji_list = [c for c in text if c in emoji.UNICODE_EMOJI]
        emojis = ' '.join(emoji_list)

        for emoji_ in emoji_list:
            text = re.sub(emoji_, '', text)

        return text, emojis

    def extract_emoticons(self, text):
        desc = emot.emoticons(text)
        try:  # there's a bug in 'emot' library causing TypeError in some cases
            emoticons = ' '.join(desc['value'])

            for emoticon in set([''.join(f'\{x}' for x in v) for v in desc['value']]):
                try:
                    text = re.sub(emoticon, '', text)
                except Exception:
                    pass
        except TypeError:
            emoticons, emoticon_text, emoticon_pos = '', '', ''

        return text, emoticons

    def full_sanitization(self, text):
        text = self.partial_sanitization(text)

        text = re.sub(self.hash_pat, '', text)

        text, emojis = self.extract_emoji(text)
        text, emoticons = self.extract_emoticons(text)

        text = re.sub(self.space_pat, ' ', text)

        return text, emojis, emoticons

    def sanitize_tweets(self, in_file, out_file, full_sanitize=False, reduce_to_polish=True, save_texts_only=False):
        with open(in_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            # 10-th column is a tweet to sanitize; 11-th column is tweet language
            with open(out_file, 'w') as wf:
                writer = csv.writer(wf)
                if save_texts_only:
                    writer.writerow(['tweet'])
                else:
                    if full_sanitize:
                        writer.writerow(header[:11] + ['emojis', 'emoticons'] + header[11:])
                    else:
                        writer.writerow(header)

                for row in reader:
                    if len(row) > 35 and (not reduce_to_polish or row[11] == 'pl' or row[12] == 'pl'):
                        if save_texts_only:
                            if full_sanitize:
                                text, _, _ = self.full_sanitization(row[10])
                                writer.writerow([text])
                            else:
                                text = self.partial_sanitization(row[10])
                                writer.writerow([text])
                        else:
                            if full_sanitize:
                                text, emojis, emoticons = self.full_sanitization(row[10])
                                writer.writerow(row[:10] + [text, emojis, emoticons] + row[11:])
                            else:
                                text = self.partial_sanitization(row[10])
                                writer.writerow(row[:10] + [text] + row[11:])