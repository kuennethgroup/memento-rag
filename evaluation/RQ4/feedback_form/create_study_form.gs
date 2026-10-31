/**
 * Memento user study: questionnaire at the end of each session (RQ4, moderated sessions).
 *
 * Creates the Google Form in your Google Drive:
 *   1. open https://script.google.com → New project, paste this file, save;
 *   2. choose createMementoStudyForm → Run, allow the Forms permission;
 *   3. View → Logs (Execution log) shows the edit link and the link to give the participant.
 * Running it again creates a second, separate form.
 *
 * Not anonymous like the trial form: the participant code joins the answers with the chats and the observation
 * sheet. No name or e-mail is collected. The Memento items M1–M8 use the trial v2 script's wording; the trial form
 * as sent had shorter wording of the same statements, so the two compare item by item.
 * The SUS is the all-positive version (Sauro & Lewis, 2011): score = sum of (response − 1) × 2.5, with no
 * reverse-coded items; each SUS statement is a 1–5 scale, so the response is the number itself.
 */

const CODES = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06'];

const AGREE = ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'];

const SUS = [
  'I think that I would like to use Memento frequently.',
  'I found Memento to be simple.',
  'I thought Memento was easy to use.',
  'I think that I could use Memento without the support of a technical person.',
  'I found the various functions in Memento were well integrated.',
  'I thought there was a lot of consistency in Memento.',
  'I would imagine that most people would learn to use Memento very quickly.',
  'I found Memento very intuitive.',
  'I felt very confident using Memento.',
  'I could use Memento without having to learn anything new.',
];

// [statement, optional]: only M5 and M6 depend on something a participant may not have tried; left empty means
// "didn't try" (the trial form's "Didn't try / can't say" column).
const MEMENTO = [
  ['I could find the information I was looking for quickly.', false],                      // M1 locating knowledge
  ['As far as I could judge, the answers were correct.', false],                           // M2 answer quality
  ['The linked sources made it easy to check an answer.', false],                          // M3 provenance
  ['Seeing the sources made me trust the answers more.', false],                           // M4 trust
  ['Answers about figures and tables were useful.', true],                                 // M5 multimodality
  ['Asking follow-up questions in the same chat worked well.', true],                      // M6 multi-turn
  ['Compared to how I usually search papers and theses, this was faster.', false],         // M7 comparison
  ['Memento would help a new group member get to know the group\'s previous work.', false], // M8 knowledge persistence
];

const ROLES = ["Master's student", 'PhD student', 'Postdoc', 'Other'];
const TIME_AT_CHAIR = ['Less than 1 year', '1 to 3 years', 'More than 3 years'];
const USUAL_SEARCH = ['Google Scholar or a web search', 'The university library catalogue',
                      'The chair\'s shared folders or drives', 'Asking colleagues or my supervisor',
                      'Reading earlier dissertations directly'];

function createMementoStudyForm() {
  const form = FormApp.create('Memento user study: questionnaire');
  form.setDescription(
    'Thank you for taking part! This takes about 5 minutes. Your answers are stored under your participant code, ' +
    'not your name, and reported anonymously.');
  form.setProgressBar(true);
  form.setAllowResponseEdits(false);
  form.setShowLinkToRespondAgain(false);

  form.addListItem()
    .setTitle('Participant code')
    .setHelpText('Your moderator tells you your code.')
    .setChoiceValues(CODES)
    .setRequired(true);

  form.addCheckboxItem()
    .setTitle('Consent')
    .setChoiceValues(['I agree that my answers in this form, my chats in Memento and the session notes are used ' +
                      'for the thesis, stored under my participant code and reported anonymously.'])
    .setRequired(true);

  form.addPageBreakItem().setTitle('About you');

  form.addMultipleChoiceItem()
    .setTitle('Your role at the chair')
    .setChoiceValues(ROLES)
    .setRequired(true);

  form.addMultipleChoiceItem()
    .setTitle('How long have you been at the chair?')
    .setChoiceValues(TIME_AT_CHAIR)
    .setRequired(true);

  form.addCheckboxItem()
    .setTitle('How do you usually look for earlier work on a topic?')
    .setHelpText('Choose all that apply.')
    .setChoiceValues(USUAL_SEARCH)
    .showOtherOption(true)
    .setRequired(true);

  // One 1–5 scale per SUS statement rather than a grid: a grid squeezes the statements into a narrow column.
  form.addPageBreakItem()
    .setTitle('Using Memento')
    .setHelpText('How much do you agree with each statement? 1 = strongly disagree, 5 = strongly agree.');

  SUS.forEach(function (statement) {
    form.addScaleItem()
      .setTitle(statement)
      .setBounds(1, 5)
      .setLabels(AGREE[0], AGREE[AGREE.length - 1])
      .setRequired(true);
  });

  form.addPageBreakItem()
    .setTitle('Memento for your work')
    .setHelpText('How much do you agree with each statement? 1 = strongly disagree, 5 = strongly agree.');

  MEMENTO.forEach(function (item) {
    const scale = form.addScaleItem()
      .setTitle(item[0])
      .setBounds(1, 5)
      .setLabels(AGREE[0], AGREE[AGREE.length - 1])
      .setRequired(!item[1]);
    if (item[1]) scale.setHelpText('Leave empty if you did not try this.');
  });

  form.addParagraphTextItem()
    .setTitle('Anything else you would like to tell me?')
    .setHelpText('Optional.')
    .setRequired(false);

  form.setConfirmationMessage('Thank you very much for taking part!');

  // Forms created by script may start unpublished (newer Google Forms publishing settings).
  if (typeof form.setPublished === 'function') form.setPublished(true);

  Logger.log('Edit the form:            ' + form.getEditUrl());
  Logger.log('Link for the participant: ' + form.getPublishedUrl());
}
