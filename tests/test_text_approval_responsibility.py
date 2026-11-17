"""Selected approval and responsibility cues must survive local rewriting."""

import pytest

from remove_ai_watermarks.text import verify_text_rewrite_facts


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Staff may extend the program if funding is approved.", "Staff may extend it if funding is secured."),
        ("Staff may extend it if funding is secured.", "Staff may extend the program if funding is approved."),
        ("Work begins after the budget is authorized.", "Work begins after the budget is available."),
        ("Borrowers do not repair the tools.", "Borrowers are not responsible for repairing any tools."),
        ("Borrowers are not responsible for repairs.", "Borrowers do not perform repairs."),
        ("Staff inspect the toolkit.", "Staff are accountable for inspecting the toolkit."),
        ("Staff are in charge of inspecting the toolkit.", "Staff inspect the toolkit."),
        ("Работу продолжат, если финансирование одобрено.", "Работу продолжат, если финансирование обеспечено."),
        ("Работу продолжат, если финансирование обеспечено.", "Работу продолжат, если финансирование одобрено."),
        ("Работа начнётся, когда бюджет утвердят.", "Работа начнётся, когда бюджет появится."),
        ("Заёмщики не ремонтируют инструменты.", "Заёмщики не несут ответственности за ремонт инструментов."),
        ("Заёмщики не несут ответственности за ремонт.", "Заёмщики не выполняют ремонт."),
        ("Сотрудники осматривают набор.", "Сотрудники отвечают за осмотр набора."),
        ("Borrowers do not repair tools.", "Borrowers are not allowed to repair tools."),
        ("Borrowers are not permitted to repair tools.", "Borrowers do not repair tools."),
        ("Заёмщики не ремонтируют инструменты.", "Заёмщикам не разрешено ремонтировать инструменты."),
        ("Заёмщикам не разрешено ремонтировать инструменты.", "Заёмщики не ремонтируют инструменты."),
    ],
)
def test_approval_and_responsibility_changes_are_rejected(source, candidate):
    assert not verify_text_rewrite_facts(source, candidate)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Staff may continue if funding is approved.", "Staff may continue if funding receives approval."),
        ("The committee granted approval for the plan.", "The committee approved the plan."),
        ("Funding was authorised.", "Funding received authorisation."),
        ("Staff are responsible for inspections.", "Responsibility for inspections rests with staff."),
        ("Staff have responsibility for inspections.", "Staff are responsible for inspections."),
        ("Borrowers do not repair tools.", "Borrowers don't repair tools."),
        ("Бюджет утвердили.", "Бюджет был утверждён."),
        ("Финансирование одобрено.", "Финансирование получило одобрение."),
        ("Волонтёры несут ответственность за осмотр набора.", "Волонтёры отвечают за осмотр набора."),
        ("Заёмщики не ремонтируют инструменты.", "Заёмщики не занимаются ремонтом инструментов."),
        ("Borrowers are allowed to repair tools.", "Borrowers are permitted to repair tools."),
        ("Заёмщикам разрешили ремонт.", "Заёмщики получили разрешение на ремонт."),
    ],
)
def test_selected_equivalent_forms_remain_accepted(source, candidate):
    assert verify_text_rewrite_facts(source, candidate)
