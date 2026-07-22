from app.asr_number_normalizer import normalize_asr_numbers
from app.transcript_postprocessor import TranscriptPostProcessor


def test_required_number_formats():
    cases = {
        "twenty thirty four": "2034",
        "twelve forty three": "1243",
        "fifteen thirty eight": "1538",
        "double five": "55",
        "triple five double nine four zero": "5559940",
        "two thousand thirty four": "2034",
        "twelve hundred thirty four": "1234",
    }
    for source, expected in cases.items():
        assert normalize_asr_numbers(source) == expected


def test_ordinary_numbers_are_not_changed():
    assert normalize_asr_numbers("about two days ago") == "about two days ago"
    assert normalize_asr_numbers("one to three business days") == "one to three business days"
    assert normalize_asr_numbers("it is twenty five dollars") == "it is twenty five dollars"
    assert normalize_asr_numbers("one hundred two and three") == "one hundred two and three"


def test_cross_turn_context_and_entity_guard():
    processor = TranscriptPostProcessor()
    assert processor.process("Thank you for calling Inspire Financial", is_final=True).text.endswith(
        "Inspira Financial"
    )
    assert processor.process("Can I get your member ID?", is_final=True).text.endswith("member ID?")
    assert processor.process("Sure it is twenty forty three", is_final=True).text.endswith("2043")
    assert processor.process("I want to inspire financial confidence", is_final=True).text == (
        "I want to inspire financial confidence"
    )
