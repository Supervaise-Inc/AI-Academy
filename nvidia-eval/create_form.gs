/**
 * Supervaise x NVIDIA AI Academy - 3-Day NVIDIA Training Evaluation
 * Run createEvaluationForm() once from script.google.com. Logs the edit + share links.
 */
function createEvaluationForm() {
  var form = FormApp.create('3-Day NVIDIA Training Evaluation | Supervaise AI Academy');
  form.setDescription(
    'Thank you for joining the 3-day NVIDIA training, delivered by Supervaise Inc. in partnership with NVIDIA.\n' +
    'Your honest feedback helps us improve future AI Academy sessions. This takes about 2 minutes.\n\n' +
    'Unlocking Possibilities Together');
  form.setProgressBar(true).setShowLinkToRespondAgain(false)
      .setConfirmationMessage('Thank you! Your feedback helps us build a better Supervaise AI Academy.');

  // Header banner with Supervaise logo (upload form_header.png to Drive and put its file ID here, or set it via Customize theme)
  var HEADER_FILE_ID = ''; // optional
  if (HEADER_FILE_ID) {
    form.addImageItem().setImage(DriveApp.getFileById(HEADER_FILE_ID).getBlob()).setAlignment(FormApp.Alignment.CENTER);
  }

  var agree = ['1 - Poor', '2', '3', '4', '5 - Excellent'];
  function scale(title, help, low, high) {
    form.addScaleItem().setTitle(title).setHelpText(help).setBounds(1, 5).setLabels(low, high).setRequired(true);
  }

  scale('1. How would you rate the training overall?', 'Across all 3 days.', 'Poor', 'Excellent');
  scale('2. How relevant was the content to your work or studies?', 'Did the topics apply to what you do?', 'Not relevant', 'Highly relevant');
  scale('3. How clear and effective was the instructor/trainer?', 'Explanations, pacing, and answers to questions.', 'Unclear', 'Very clear');
  scale('4. How useful were the hands-on labs and demos on NVIDIA compute?', 'Practical sessions and exercises.', 'Not useful', 'Very useful');
  scale('5. How confident are you now in applying what you learned?', 'Compared with before the training.', 'Not confident', 'Very confident');

  form.addMultipleChoiceItem()
      .setTitle('6. How did you find the pace and structure of the 3 days?')
      .setChoiceValues(['Too slow', 'Slightly slow', 'Just right', 'Slightly fast', 'Too fast'])
      .setRequired(true);

  form.addParagraphTextItem()
      .setTitle('7. Comments and suggestions')
      .setHelpText('What did you enjoy most, and what should we improve or add in future trainings? (optional)')
      .setRequired(false);

  Logger.log('Edit: ' + form.getEditUrl());
  Logger.log('Share: ' + form.getPublishedUrl());
}
